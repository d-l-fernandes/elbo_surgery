import tensorflow as tf

from mvg_dist.linear_operators import LinearOperatorLogDiag, LinearOperatorLogIdentity
from tensorflow_probability.python.internal import dtype_util


def _convert_to_tensor(x, name, dtype=None):
    return None if x is None else tf.convert_to_tensor(x, name=name, dtype=dtype)


def make_log_diag_scale(loc=None,
                        log_covariance_diag=None,
                        shape_hint=None,
                        validate_args=False,
                        assert_positive=False,
                        name=None,
                        dtype=None):

    with tf.name_scope(
            name,
            "make_log_diag_scale",
            values=[loc, log_covariance_diag]):
        if dtype is None:
            dtype = dtype_util.common_dtype(
                [loc, log_covariance_diag],
                preferred_dtype=tf.float32)
        loc = _convert_to_tensor(loc, name="loc", dtype=dtype)
        log_covariance_diag = _convert_to_tensor(log_covariance_diag, name="log_covariance_diag", dtype=dtype)

        if log_covariance_diag is not None:
            return LinearOperatorLogDiag(
                diag=log_covariance_diag / 2.,
                is_non_singular=True,
                is_self_adjoint=True,
                is_positive_definite=assert_positive)

        if loc is None and shape_hint is None:
            raise ValueError("Cannot infer `event_shape` unless `loc` or "
                             "`shape_hint` is specified.")

        num_rows = shape_hint
        del shape_hint
        if num_rows is None:
            num_rows = tf.dimension_value(loc.shape[-1])
            if num_rows is None:
                num_rows = tf.shape(loc)[-1]

        return LinearOperatorLogIdentity(
            num_rows=num_rows,
            is_non_singular=True,
            is_self_adjoint=True,
            is_positive_definite=True,
            assert_proper_shapes=validate_args)


def is_diagonal_scale(scale):
    """Returns `True` if `scale` is a `LinearOperator` that is known to be diag.
    Args:
      scale:  `LinearOperator` instance.
    Returns:
      Python `bool`.
    Raises:
      TypeError:  If `scale` is not a `LinearOperator`.
    """
    if not isinstance(scale, tf.linalg.LinearOperator):
        raise TypeError("Expected argument 'scale' to be instance of LinearOperator"
                        ". Found: %s" % scale)
    return (isinstance(scale, tf.linalg.LinearOperatorIdentity) or
            isinstance(scale, tf.linalg.LinearOperatorScaledIdentity) or
            isinstance(scale, tf.linalg.LinearOperatorDiag) or
            isinstance(scale, LinearOperatorLogDiag))
