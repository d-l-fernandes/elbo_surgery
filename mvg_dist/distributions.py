import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import affine_linear_operator as affine_linear_operator_bijector
from tensorflow_probability.python.distributions import kullback_leibler
from tensorflow_probability.python.distributions import normal
from tensorflow_probability.python.distributions import transformed_distribution
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util

from mvg_dist import distribution_util as custom_dist_util
from mvg_dist import linear_operators as custom_lin_op

_mvn_sample_note = """
`value` is a batch vector with compatible shape if `value` is a `Tensor` whose
shape can be broadcast up to either:
```python
self.batch_shape + self.event_shape
```
or
```python
[M1, ..., Mm] + self.batch_shape + self.event_shape
```
"""


class MultivariateNormalLogLinearOperator(transformed_distribution.TransformedDistribution):

    def __init__(self,
                 loc=None,
                 scale=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="MultivariateNormalLinearOperator"):
        parameters = dict(locals())
        if scale is None:
            raise ValueError("Missing required `scale` parameter.")
        if not scale.dtype.is_floating:
            raise TypeError("`scale` parameter must have floating-point dtype.")

        with tf.name_scope(name, values=[loc] + scale.graph_parents) as name:
            # Since expand_dims doesn't preserve constant-ness, we obtain the
            # non-dynamic value if possible.
            loc = loc if loc is None else tf.convert_to_tensor(
                loc, name="loc", dtype=scale.dtype)
            batch_shape, event_shape = distribution_util.shapes_from_loc_and_scale(
                loc, scale)

        super(MultivariateNormalLogLinearOperator, self).__init__(
            distribution=normal.Normal(
                loc=tf.zeros([], dtype=scale.dtype),
                scale=tf.ones([], dtype=scale.dtype)),
            bijector=affine_linear_operator_bijector.AffineLinearOperator(
                shift=loc, scale=scale, validate_args=validate_args),
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
            name=name)
        self._parameters = parameters

    @property
    def loc(self):
        """The `loc` `Tensor` in `Y = scale @ X + loc`."""
        return self.bijector.shift

    @property
    def scale(self):
        """The `scale` `LinearOperator` in `Y = scale @ X + loc`."""
        return self.bijector.scale

    @distribution_util.AppendDocstring(_mvn_sample_note)
    def _log_prob(self, x):
        return super(MultivariateNormalLogLinearOperator, self)._log_prob(x)

    @distribution_util.AppendDocstring(_mvn_sample_note)
    def _prob(self, x):
        return super(MultivariateNormalLogLinearOperator, self)._prob(x)

    def _mean(self):
        shape = self.batch_shape.concatenate(self.event_shape)
        has_static_shape = shape.is_fully_defined()
        if not has_static_shape:
            shape = tf.concat([
                self.batch_shape_tensor(),
                self.event_shape_tensor(),
            ], 0)

        if self.loc is None:
            return tf.zeros(shape, self.dtype)

        if has_static_shape and shape == self.loc.shape:
            return tf.identity(self.loc)

        # Add dummy tensor of zeros to broadcast.  This is only necessary if shape
        # != self.loc.shape, but we could not determine if this is the case.
        return tf.identity(self.loc) + tf.zeros(shape, self.dtype)

    def _covariance(self):
        if custom_dist_util.is_diagonal_scale(self.scale):
            return tf.square(tf.matrix_diag(self.scale.diag_part()))
        else:
            return self.scale.matmul(self.scale.to_dense(), adjoint_arg=True)

    def _variance(self):
        if custom_dist_util.is_diagonal_scale(self.scale):
            return tf.square(self.scale.diag_part())
        elif (isinstance(self.scale, tf.linalg.LinearOperatorLowRankUpdate) and
              self.scale.is_self_adjoint):
            return tf.matrix_diag_part(self.scale.matmul(self.scale.to_dense()))
        else:
            return tf.matrix_diag_part(
                self.scale.matmul(self.scale.to_dense(), adjoint_arg=True))

    def _stddev(self):
        if custom_dist_util.is_diagonal_scale(self.scale):
            return tf.abs(self.scale.diag_part())
        elif (isinstance(self.scale, tf.linalg.LinearOperatorLowRankUpdate) and
              self.scale.is_self_adjoint):
            return tf.sqrt(
                tf.matrix_diag_part(self.scale.matmul(self.scale.to_dense())))
        else:
            return tf.sqrt(
                tf.matrix_diag_part(
                    self.scale.matmul(self.scale.to_dense(), adjoint_arg=True)))

    def _mode(self):
        return self._mean()


@kullback_leibler.RegisterKL(MultivariateNormalLogLinearOperator,
                             MultivariateNormalLogLinearOperator)
def _kl_brute_force(a, b, name=None):
    """Batched KL divergence `KL(a || b)` for multivariate Normals.
    With `X`, `Y` both multivariate Normals in `R^k` with means `mu_a`, `mu_b` and
    covariance `C_a`, `C_b` respectively,
    ```
    KL(a || b) = 0.5 * ( L - k + T + Q ),
    L := Log[Det(C_b)] - Log[Det(C_a)]
    T := trace(C_b^{-1} C_a),
    Q := (mu_b - mu_a)^T C_b^{-1} (mu_b - mu_a),
    ```
    This `Op` computes the trace by solving `C_b^{-1} C_a`. Although efficient
    methods for solving systems with `C_b` may be available, a dense version of
    (the square root of) `C_a` is used, so performance is `O(B s k**2)` where `B`
    is the batch size, and `s` is the cost of solving `C_b x = y` for vectors `x`
    and `y`.
    Args:
      a: Instance of `MultivariateNormalLinearOperator`.
      b: Instance of `MultivariateNormalLinearOperator`.
      name: (optional) name to use for created ops. Default "kl_mvn".
    Returns:
      Batchwise `KL(a || b)`.
    """

    def squared_frobenius_norm(x):
        """Helper to make KL calculation slightly more readable."""
        # http://mathworld.wolfram.com/FrobeniusNorm.html
        # The gradient of KL[p,q] is not defined when p==q. The culprit is
        # tf.norm, i.e., we cannot use the commented out code.
        # return tf.square(tf.norm(x, ord="fro", axis=[-2, -1]))
        return tf.reduce_sum(tf.square(x), axis=[-2, -1])

    def is_diagonal(x):
        """Helper to identify if `LinearOperator` has only a diagonal component."""
        return (isinstance(x, tf.linalg.LinearOperatorIdentity) or
                isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
                isinstance(x, tf.linalg.LinearOperatorDiag) or
                isinstance(x, custom_lin_op.LinearOperatorLogDiag) or
                isinstance(x, custom_lin_op.LinearOperatorLogIdentity))

    with tf.name_scope(
            name,
            "kl_mvn",
            values=[a.loc, b.loc] + a.scale.graph_parents + b.scale.graph_parents):
        # Calculation is based on:
        # http://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        # and,
        # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
        # i.e.,
        #   If Ca = AA', Cb = BB', then
        #   tr[inv(Cb) Ca] = tr[inv(B)' inv(B) A A']
        #                  = tr[inv(B) A A' inv(B)']
        #                  = tr[(inv(B) A) (inv(B) A)']
        #                  = sum_{ij} (inv(B) A)_{ij}**2
        #                  = ||inv(B) A||_F**2
        # where ||.||_F is the Frobenius norm and the second equality follows from
        # the cyclic permutation property.
        if is_diagonal(a.scale) and is_diagonal(b.scale):
            # Using `stddev` because it handles expansion of Identity cases.
            b_inv_a = (a.stddev() / b.stddev())[..., tf.newaxis]
        else:
            b_inv_a = b.scale.solve(a.scale.to_dense())
        kl_div = (
                b.scale.log_abs_determinant() - a.scale.log_abs_determinant() +
                0.5 * (-tf.cast(a.scale.domain_dimension_tensor(), a.dtype) +
                       squared_frobenius_norm(b_inv_a) + squared_frobenius_norm(
                    b.scale.solve((b.mean() - a.mean())[..., tf.newaxis]))))
        kl_div.set_shape(tf.broadcast_static_shape(a.batch_shape, b.batch_shape))
        return kl_div

@kullback_leibler.RegisterKL(tfp.distributions.MultivariateNormalLinearOperator,
                             MultivariateNormalLogLinearOperator)
def _kl_brute_force2(a, b, name=None):
    return _kl_brute_force(a, b, name)


@kullback_leibler.RegisterKL(MultivariateNormalLogLinearOperator,
                             tfp.distributions.MultivariateNormalLinearOperator,
                             )
def _kl_brute_force3(a, b, name=None):
    return _kl_brute_force(a, b, name)


class MultivariateNormalLogDiag(MultivariateNormalLogLinearOperator):

    def __init__(self,
                 loc=None,
                 log_covariance_diag=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="MultivariateNormalDiag"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            with tf.name_scope(
                    "init", values=[loc, log_covariance_diag]):
                # No need to validate_args while making diag_scale.  The returned
                # LinearOperatorDiag has an assert_non_singular method that is called by
                # the Bijector.
                scale = custom_dist_util.make_log_diag_scale(
                    loc=loc,
                    log_covariance_diag=log_covariance_diag,
                    validate_args=False,
                    assert_positive=False)
        super(MultivariateNormalLogDiag, self).__init__(
            loc=loc,
            scale=scale,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)
        self._parameters = parameters


class MultivariateNormalTriLLogDiagonal(MultivariateNormalLogLinearOperator):
    def __init__(self,
                 loc=None,
                 scale_tril_log_diag=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 name="MultivariateNormalTriL"):
        parameters = dict(locals())

        def _convert_to_tensor(x, name, dtype):
            return None if x is None else tf.convert_to_tensor(
                x, name=name, dtype=dtype)

        if loc is None and scale_tril_log_diag is None:
            raise ValueError("Must specify one or both of `loc`, `scale_tril`.")
        with tf.name_scope(name) as name:
            with tf.name_scope("init", values=[loc, scale_tril_log_diag]):
                dtype = dtype_util.common_dtype([loc, scale_tril_log_diag], tf.float32)
                loc = _convert_to_tensor(loc, name="loc", dtype=dtype)
                scale_tril_log_diag = _convert_to_tensor(
                    scale_tril_log_diag, name="scale_tril", dtype=dtype)
                if scale_tril_log_diag is None:
                    scale = custom_lin_op.LinearOperatorLogIdentity(
                        num_rows=distribution_util.dimension_size(loc, -1),
                        dtype=loc.dtype,
                        is_self_adjoint=True,
                        is_positive_definite=True,
                        assert_proper_shapes=validate_args)
                else:
                    # No need to validate that scale_tril is non-singular.
                    # LinearOperatorLowerTriangular has an assert_non_singular
                    # method that is called by the Bijector.
                    scale = custom_lin_op.LinearOperatorLowerTriangularLogDiagonal(
                        scale_tril_log_diag,
                        is_non_singular=True,
                        is_self_adjoint=False,
                        is_positive_definite=False)
        super(MultivariateNormalTriLLogDiagonal, self).__init__(
            loc=loc,
            scale=scale,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)
        self._parameters = parameters


class InverseGamma(tfp.distributions.InverseGamma):
    """InverseGamma distribution where the log_prob can be evaluated with a log_x value, avoids doing log(exp(log_x))
       to get the log(x) value needed for the log_prob """

    def _log_prob(self, log_x):
        return self._log_unnormalized_prob(log_x) - self._log_normalization()

    def _log_unnormalized_prob(self, log_x):
        return -(self.concentration + 1.) * log_x - self.rate / tf.exp(log_x)


class Gamma(tfp.distributions.Gamma):
    """Gamma distribution where the log_prob can be evaluated with a log_x value, avoids doing log(exp(log_x))
       to get the log(x) value needed for the log_prob """

    def _log_prob(self, log_x):
        return self._log_unnormalized_prob(log_x) - self._log_normalization()

    def _log_unnormalized_prob(self, log_x):
        return (self.concentration - 1.) * log_x - self.rate * tf.exp(log_x)
