import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util


class LinearOperatorLogDiag(linear_operator.LinearOperator):

    def __init__(self,
                 diag,
                 is_non_singular=None,
                 is_self_adjoint=None,
                 is_positive_definite=None,
                 is_square=None,
                 name="LinearOperatorLogDiag"):

        with ops.name_scope(name, values=[diag]):
            self._diag = ops.convert_to_tensor(diag, name="diag")
            self._check_diag(self._diag)

            # Check and auto-set hints.
            if not self._diag.dtype.is_complex:
                if is_self_adjoint is False:
                    raise ValueError("A real diagonal operator is always self adjoint.")
                else:
                    is_self_adjoint = True

            if is_square is False:
                raise ValueError("Only square diagonal operators currently supported.")
            is_square = True

            super(LinearOperatorLogDiag, self).__init__(
                dtype=self._diag.dtype,
                graph_parents=[self._diag],
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                is_square=is_square,
                name=name)

    @staticmethod
    def _check_diag(diag):
        """Static check of diag."""
        allowed_dtypes = [
            dtypes.float16,
            dtypes.float32,
            dtypes.float64,
            dtypes.complex64,
            dtypes.complex128,
        ]

        dtype = diag.dtype
        if dtype not in allowed_dtypes:
            raise TypeError(
                "Argument diag must have dtype in %s.  Found: %s"
                % (allowed_dtypes, dtype))

        if diag.get_shape().ndims is not None and diag.get_shape().ndims < 1:
            raise ValueError("Argument diag must have at least 1 dimension.  "
                             "Found: %s" % diag)

    def _shape(self):
        # If d_shape = [5, 3], we return [5, 3, 3].
        d_shape = self._diag.get_shape()
        return d_shape.concatenate(d_shape[-1:])

    def _shape_tensor(self):
        d_shape = array_ops.shape(self._diag)
        k = d_shape[-1]
        return array_ops.concat((d_shape, [k]), 0)

    def _assert_non_singular(self):
        return linear_operator_util.assert_no_entries_with_modulus_zero(
            math_ops.exp(self._diag),
            message="Singular operator:  Diagonal contained zero values.")

    def _assert_positive_definite(self):
        if self.dtype.is_complex:
            message = (
                "Diagonal operator had diagonal entries with non-positive real part, "
                "thus was not positive definite.")
        else:
            message = (
                "Real diagonal operator had non-positive diagonal entries, "
                "thus was not positive definite.")

        return check_ops.assert_positive(
            math_ops.real(math_ops.exp(self._diag)),
            message=message)

    def _assert_self_adjoint(self):
        return linear_operator_util.assert_zero_imag_part(
            math_ops.exp(self._diag),
            message=(
                "This diagonal operator contained non-zero imaginary values.  "
                " Thus it was not self-adjoint."))

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        diag_term = math_ops.conj(math_ops.exp(self._diag)) if adjoint else math_ops.exp(self._diag)
        x = linalg.adjoint(x) if adjoint_arg else x
        diag_mat = array_ops.expand_dims(diag_term, -1)
        return diag_mat * x

    def _determinant(self):
        return math_ops.reduce_prod(math_ops.exp(self._diag), axis=[-1])

    def _log_abs_determinant(self):
        log_det = math_ops.reduce_sum(self._diag, axis=[-1])
        if self.dtype.is_complex:
            log_det = math_ops.cast(log_det, dtype=self.dtype)
        return log_det

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        diag_term = math_ops.conj(math_ops.exp(self._diag)) if adjoint else math_ops.exp(self._diag)
        rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
        inv_diag_mat = array_ops.expand_dims(1. / diag_term, -1)
        return rhs * inv_diag_mat

    def _to_dense(self):
        return array_ops.matrix_diag(math_ops.exp(self._diag))

    def _diag_part(self):
        return self.diag

    def _add_to_tensor(self, x):
        x_diag = array_ops.matrix_diag_part(x)
        new_diag = math_ops.exp(self._diag) + x_diag
        return array_ops.matrix_set_diag(x, new_diag)

    @property
    def diag(self):
        return math_ops.exp(self._diag)


class LinearOperatorLowerTriangularLogDiagonal(linear_operator.LinearOperator):

    def __init__(self,
                 tril,
                 is_non_singular=None,
                 is_self_adjoint=None,
                 is_positive_definite=None,
                 is_square=None,
                 name="LinearOperatorLowerTriangular"):

        if is_square is False:
            raise ValueError(
                "Only square lower triangular operators supported at this time.")
        is_square = True

        with ops.name_scope(name, values=[tril]):
            self._tril = ops.convert_to_tensor(tril, name="tril")
            self._check_tril(self._tril)
            self._tril = array_ops.matrix_band_part(tril, -1, 0)
            self._diag = array_ops.matrix_diag_part(self._tril)

            super(LinearOperatorLowerTriangularLogDiagonal, self).__init__(
                dtype=self._tril.dtype,
                graph_parents=[self._tril],
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                is_square=is_square,
                name=name)

    @staticmethod
    def _check_tril(tril):
        """Static check of the `tril` argument."""
        allowed_dtypes = [
            dtypes.float16,
            dtypes.float32,
            dtypes.float64,
            dtypes.complex64,
            dtypes.complex128,
        ]
        dtype = tril.dtype
        if dtype not in allowed_dtypes:
            raise TypeError(
                "Argument tril must have dtype in %s.  Found: %s"
                % (allowed_dtypes, dtype))

        if tril.get_shape().ndims is not None and tril.get_shape().ndims < 2:
            raise ValueError(
                "Argument tril must have at least 2 dimensions.  Found: %s"
                % tril)

    def _shape(self):
        return self._tril.get_shape()

    def _shape_tensor(self):
        return array_ops.shape(self._tril)

    def _assert_non_singular(self):
        return linear_operator_util.assert_no_entries_with_modulus_zero(
            math_ops.exp(self._diag),
            message="Singular operator:  Diagonal contained zero values.")

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        return linear_operator_util.matmul_with_broadcast(
            array_ops.matrix_set_diag(self._tril, math_ops.exp(self._diag)), x,
            adjoint_a=adjoint, adjoint_b=adjoint_arg)

    def _determinant(self):
        return math_ops.reduce_prod(math_ops.exp(self._diag), axis=[-1])

    def _log_abs_determinant(self):
        return math_ops.reduce_sum(self._diag, axis=[-1])

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
        return linear_operator_util.matrix_triangular_solve_with_broadcast(
            array_ops.matrix_set_diag(self._tril, math_ops.exp(self._diag)), rhs, lower=True, adjoint=adjoint)

    def _to_dense(self):
        return array_ops.matrix_set_diag(self._tril, math_ops.exp(self._diag))

    def _add_to_tensor(self, x):
        return array_ops.matrix_set_diag(self._tril, math_ops.exp(self._diag)) + x


class BaseLinearOperatorLogIdentity(linear_operator.LinearOperator):
    """Base class for Log Identity operators."""

    def _check_num_rows_possibly_add_asserts(self):
        """Static check of init arg `num_rows`, possibly add asserts."""
        # Possibly add asserts.
        if self._assert_proper_shapes:
            self._num_rows = control_flow_ops.with_dependencies([
                check_ops.assert_rank(
                    self._num_rows,
                    0,
                    message="Argument num_rows must be a 0-D Tensor."),
                check_ops.assert_non_negative(
                    self._num_rows,
                    message="Argument num_rows must be non-negative."),
            ], self._num_rows)

        # Static checks.
        if not self._num_rows.dtype.is_integer:
            raise TypeError("Argument num_rows must be integer type.  Found:"
                            " %s" % self._num_rows)

        num_rows_static = self._num_rows_static

        if num_rows_static is None:
            return  # Cannot do any other static checks.

        if num_rows_static.ndim != 0:
            raise ValueError("Argument num_rows must be a 0-D Tensor.  Found:"
                             " %s" % num_rows_static)

        if num_rows_static < 0:
            raise ValueError("Argument num_rows must be non-negative.  Found:"
                             " %s" % num_rows_static)

    def _min_matrix_dim(self):
        """Minimum of domain/range dimension, if statically available, else None."""
        domain_dim = tensor_shape.dimension_value(self.domain_dimension)
        range_dim = tensor_shape.dimension_value(self.range_dimension)
        if domain_dim is None or range_dim is None:
            return None
        return min(domain_dim, range_dim)

    def _min_matrix_dim_tensor(self):
        """Minimum of domain/range dimension, as a tensor."""
        return math_ops.reduce_min(self.shape_tensor()[-2:])

    def _zeros_diag(self):
        """Returns the diagonal of this operator as all zeros."""
        if self.shape.is_fully_defined():
            d_shape = self.batch_shape.concatenate([self._min_matrix_dim()])
        else:
            d_shape = array_ops.concat(
                [self.batch_shape_tensor(),
                 [self._min_matrix_dim_tensor()]], axis=0)

        return array_ops.zeros(shape=d_shape, dtype=self.dtype)


class LinearOperatorLogIdentity(BaseLinearOperatorLogIdentity):

    def __init__(self,
                 num_rows,
                 batch_shape=None,
                 dtype=None,
                 is_non_singular=True,
                 is_self_adjoint=True,
                 is_positive_definite=True,
                 is_square=True,
                 assert_proper_shapes=False,
                 name="LinearOperatorLogIdentity"):
        dtype = dtype or dtypes.float32
        self._assert_proper_shapes = assert_proper_shapes

        with ops.name_scope(name):
            dtype = dtypes.as_dtype(dtype)
            if not is_self_adjoint:
                raise ValueError("An identity operator is always self adjoint.")
            if not is_non_singular:
                raise ValueError("An identity operator is always non-singular.")
            if not is_positive_definite:
                raise ValueError("An identity operator is always positive-definite.")
            if not is_square:
                raise ValueError("An identity operator is always square.")

            super(LinearOperatorLogIdentity, self).__init__(
                dtype=dtype,
                is_non_singular=is_non_singular,
                is_self_adjoint=is_self_adjoint,
                is_positive_definite=is_positive_definite,
                is_square=is_square,
                name=name)

            self._num_rows = linear_operator_util.shape_tensor(
                num_rows, name="num_rows")
            self._num_rows_static = tensor_util.constant_value(self._num_rows)
            self._check_num_rows_possibly_add_asserts()

            if batch_shape is None:
                self._batch_shape_arg = None
            else:
                self._batch_shape_arg = linear_operator_util.shape_tensor(
                    batch_shape, name="batch_shape_arg")
                self._batch_shape_static = tensor_util.constant_value(
                    self._batch_shape_arg)
                self._check_batch_shape_possibly_add_asserts()

    def _shape(self):
        matrix_shape = tensor_shape.TensorShape((self._num_rows_static,
                                                 self._num_rows_static))
        if self._batch_shape_arg is None:
            return matrix_shape

        batch_shape = tensor_shape.TensorShape(self._batch_shape_static)
        return batch_shape.concatenate(matrix_shape)

    def _shape_tensor(self):
        matrix_shape = array_ops.stack((self._num_rows, self._num_rows), axis=0)
        if self._batch_shape_arg is None:
            return matrix_shape

        return array_ops.concat((self._batch_shape_arg, matrix_shape), 0)

    def _assert_non_singular(self):
        return control_flow_ops.no_op("assert_non_singular")

    def _assert_positive_definite(self):
        return control_flow_ops.no_op("assert_positive_definite")

    def _assert_self_adjoint(self):
        return control_flow_ops.no_op("assert_self_adjoint")

    def _possibly_broadcast_batch_shape(self, x):
        """Return 'x', possibly after broadcasting the leading dimensions."""
        # If we have no batch shape, our batch shape broadcasts with everything!
        if self._batch_shape_arg is None:
            return x

        # Static attempt:
        #   If we determine that no broadcast is necessary, pass x through
        #   If we need a broadcast, add to an array of zeros.
        #
        # special_shape is the shape that, when broadcast with x's shape, will give
        # the correct broadcast_shape.  Note that
        #   We have already verified the second to last dimension of self.shape
        #   matches x's shape in assert_compatible_matrix_dimensions.
        #   Also, the final dimension of 'x' can have any shape.
        #   Therefore, the final two dimensions of special_shape are 1's.
        special_shape = self.batch_shape.concatenate([1, 1])
        bshape = array_ops.broadcast_static_shape(x.get_shape(), special_shape)
        if special_shape.is_fully_defined():
            # bshape.is_fully_defined iff special_shape.is_fully_defined.
            if bshape == x.get_shape():
                return x
            # Use the built in broadcasting of addition.
            zeros = array_ops.zeros(shape=special_shape, dtype=self.dtype)
            return x + zeros

        # Dynamic broadcast:
        #   Always add to an array of zeros, rather than using a "cond", since a
        #   cond would require copying data from GPU --> CPU.
        special_shape = array_ops.concat((self.batch_shape_tensor(), [1, 1]), 0)
        zeros = array_ops.zeros(shape=special_shape, dtype=self.dtype)
        return x + zeros

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        # Note that adjoint has no effect since this matrix is self-adjoint.
        x = linalg.adjoint(x) if adjoint_arg else x
        if self._assert_proper_shapes:
            aps = linear_operator_util.assert_compatible_matrix_dimensions(self, x)
            x = control_flow_ops.with_dependencies([aps], x)
        return self._possibly_broadcast_batch_shape(x)

    def _determinant(self):
        return array_ops.ones(shape=self.batch_shape_tensor(), dtype=self.dtype)

    def _log_abs_determinant(self):
        return array_ops.zeros(shape=self.batch_shape_tensor(), dtype=self.dtype)

    def _solve(self, rhs, adjoint=False, adjoint_arg=False):
        return self._matmul(rhs, adjoint_arg=adjoint_arg)

    def _trace(self):
        # Get Tensor of all ones of same shape as self.batch_shape.
        if self.batch_shape.is_fully_defined():
            batch_of_ones = array_ops.ones(shape=self.batch_shape, dtype=self.dtype)
        else:
            batch_of_ones = array_ops.ones(
                shape=self.batch_shape_tensor(), dtype=self.dtype)

        if self._min_matrix_dim() is not None:
            return self._min_matrix_dim() * batch_of_ones
        else:
            return (math_ops.cast(self._min_matrix_dim_tensor(), self.dtype) *
                    batch_of_ones)

    def _diag_part(self):
        return self._zeros_diag()

    def add_to_tensor(self, mat, name="add_to_tensor"):
        """Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.
        Args:
          mat:  `Tensor` with same `dtype` and shape broadcastable to `self`.
          name:  A name to give this `Op`.
        Returns:
          A `Tensor` with broadcast shape and same `dtype` as `self`.
        """
        with self._name_scope(name, values=[mat]):
            mat = ops.convert_to_tensor(mat, name="mat")
            mat_diag = array_ops.matrix_diag_part(mat)
            new_diag = 1 + mat_diag
            return array_ops.matrix_set_diag(mat, new_diag)

    def _check_num_rows_possibly_add_asserts(self):
        """Static check of init arg `num_rows`, possibly add asserts."""
        # Possibly add asserts.
        if self._assert_proper_shapes:
            self._num_rows = control_flow_ops.with_dependencies([
                check_ops.assert_rank(
                    self._num_rows,
                    0,
                    message="Argument num_rows must be a 0-D Tensor."),
                check_ops.assert_non_negative(
                    self._num_rows,
                    message="Argument num_rows must be non-negative."),
            ], self._num_rows)

        # Static checks.
        if not self._num_rows.dtype.is_integer:
            raise TypeError("Argument num_rows must be integer type.  Found:"
                            " %s" % self._num_rows)

        num_rows_static = self._num_rows_static

        if num_rows_static is None:
            return  # Cannot do any other static checks.

        if num_rows_static.ndim != 0:
            raise ValueError("Argument num_rows must be a 0-D Tensor.  Found:"
                             " %s" % num_rows_static)

        if num_rows_static < 0:
            raise ValueError("Argument num_rows must be non-negative.  Found:"
                             " %s" % num_rows_static)

    def _check_batch_shape_possibly_add_asserts(self):
        """Static check of init arg `batch_shape`, possibly add asserts."""
        if self._batch_shape_arg is None:
            return

        # Possibly add asserts
        if self._assert_proper_shapes:
            self._batch_shape_arg = control_flow_ops.with_dependencies([
                check_ops.assert_rank(
                    self._batch_shape_arg,
                    1,
                    message="Argument batch_shape must be a 1-D Tensor."),
                check_ops.assert_non_negative(
                    self._batch_shape_arg,
                    message="Argument batch_shape must be non-negative."),
            ], self._batch_shape_arg)

        # Static checks
        if not self._batch_shape_arg.dtype.is_integer:
            raise TypeError("Argument batch_shape must be integer type.  Found:"
                            " %s" % self._batch_shape_arg)

        if self._batch_shape_static is None:
            return  # Cannot do any other static checks.

        if self._batch_shape_static.ndim != 1:
            raise ValueError("Argument batch_shape must be a 1-D Tensor.  Found:"
                             " %s" % self._batch_shape_static)

        if np.any(self._batch_shape_static < 0):
            raise ValueError("Argument batch_shape must be non-negative.  Found:"
                             "%s" % self._batch_shape_static)
