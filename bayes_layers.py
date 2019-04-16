import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import array_ops

import tensorflow_probability as tfp
from tensorflow_probability.python.layers import util as tfp_layers_util

import mvg_dist.distributions as mvg_dist

tfd = tfp.distributions


def loc_tril_covar_fn(
        is_singular=False,
        loc_initializer=tf.random_normal_initializer(stddev=0.1),
        untransformed_scale_initializer=tf.random_normal_initializer(stddev=0.1),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None):
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        """Creates `loc`, `scale` parameters."""
        loc = add_variable_fn(
            name=name + '_loc',
            shape=shape,
            initializer=loc_initializer,
            regularizer=loc_regularizer,
            constraint=loc_constraint,
            dtype=dtype,
            trainable=trainable)
        if is_singular:
            return loc, None
        tril_shape = [shape[0], shape[1] * (shape[1] + 1) // 2]
        tril_covar_log_diag = add_variable_fn(
            name=name + '_untransformed_scale',
            shape=tril_shape,
            initializer=untransformed_scale_initializer,
            regularizer=untransformed_scale_regularizer,
            constraint=untransformed_scale_constraint,
            dtype=dtype,
            trainable=trainable)
        return loc, tfd.fill_triangular(tril_covar_log_diag)
    return _fn


def loc_log_variance_fn(
        is_singular=False,
        loc_initializer=tf.random_normal_initializer(stddev=0.1),
        untransformed_scale_initializer=tf.random_normal_initializer(stddev=0.1),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None):

    def _fn(dtype, shape, name, trainable, add_variable_fn):
        """Creates `loc`, `scale` parameters."""
        loc = add_variable_fn(
            name=name + '_loc',
            shape=shape,
            initializer=loc_initializer,
            regularizer=loc_regularizer,
            constraint=loc_constraint,
            dtype=dtype,
            trainable=trainable)
        if is_singular:
            return loc, None
        untransformed_scale = add_variable_fn(
            name=name + '_untransformed_scale',
            shape=shape,
            initializer=untransformed_scale_initializer,
            regularizer=untransformed_scale_regularizer,
            constraint=untransformed_scale_constraint,
            dtype=dtype,
            trainable=trainable)
        return loc, untransformed_scale
    return _fn


def multivariate_normal_fn(
        diagonal=True,
        is_singular=False,
        loc_initializer=tf.random_normal_initializer(stddev=0.1),
        untransformed_scale_initializer=tf.random_normal_initializer(
            mean=-3., stddev=0.1),
        loc_regularizer=None,
        untransformed_scale_regularizer=None,
        loc_constraint=None,
        untransformed_scale_constraint=None):

    if diagonal:
        loc_scale_fn = loc_log_variance_fn(
            is_singular=is_singular,
            loc_initializer=loc_initializer,
            untransformed_scale_initializer=untransformed_scale_initializer,
            loc_regularizer=loc_regularizer,
            untransformed_scale_regularizer=untransformed_scale_regularizer,
            loc_constraint=loc_constraint,
            untransformed_scale_constraint=untransformed_scale_constraint)
    else:
        loc_scale_fn = loc_tril_covar_fn(
            is_singular=is_singular,
            loc_initializer=loc_initializer,
            untransformed_scale_initializer=untransformed_scale_initializer,
            loc_regularizer=loc_regularizer,
            untransformed_scale_regularizer=untransformed_scale_regularizer,
            loc_constraint=loc_constraint,
            untransformed_scale_constraint=untransformed_scale_constraint)

    def _fn(dtype, shape, name, trainable, add_variable_fn):
        loc, scale = loc_scale_fn(dtype, shape, name, trainable, add_variable_fn)
        if scale is None:
            dist = tfd.Deterministic(loc=loc)
        else:
            if diagonal:
                dist = mvg_dist.MultivariateNormalLogDiag(loc, scale)
            else:
                dist = mvg_dist.MultivariateNormalTriLLogDiagonal(loc, scale)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        final_dist = tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
        return final_dist
    return _fn


def multivariate_normal_gamma_precision_fn():

    class InverseGammaLogProb(tf.keras.regularizers.Regularizer):

        def __init__(self, prior_a=1., prior_b=0.5):
            self.dist = mvg_dist.InverseGamma(concentration=prior_a, rate=prior_b)

        def __call__(self, x):
            regularization = -tf.reduce_sum(self.dist.log_prob(x))
            return regularization

    def _fn(dtype, shape, name, trainable, add_variable_fn):
        log_alphas = add_variable_fn(
            name=name + '_log_alphas',
            shape=shape,
            initializer=tf.keras.initializers.zeros(),
            regularizer=InverseGammaLogProb(),
            constraint=None,
            dtype=dtype,
            trainable=trainable)
        dist = mvg_dist.MultivariateNormalLogDiag(tf.zeros(shape, dtype=dtype), log_alphas)
        batch_ndims = tf.size(dist.batch_shape_tensor())
        final_dist = tfd.Independent(dist, reinterpreted_batch_ndims=batch_ndims)
        return final_dist
    return _fn


class Conv2DTransposeReparameterization(tfp.layers.Convolution2DReparameterization):
    """Transposed convolutional layer (sometimes called Deconvolution)
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format='channels_last',
                 dilation_rate=(1, 1),
                 activation=None,
                 activity_regularizer=None,
                 kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
                 kernel_posterior_tensor_fn=lambda d: d.sample(),
                 kernel_prior_fn=tfp_layers_util.default_multivariate_normal_fn,
                 kernel_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
                 bias_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
                 bias_posterior_tensor_fn=lambda d: d.sample(),
                 bias_prior_fn=None,
                 bias_divergence_fn=lambda q, p, ignore: tfd.kl_divergence(q, p),
                 **kwargs):
        super(Conv2DTransposeReparameterization, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=tf.keras.activations.get(activation),
            activity_regularizer=activity_regularizer,
            kernel_posterior_fn=kernel_posterior_fn,
            kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
            kernel_prior_fn=kernel_prior_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_posterior_fn=bias_posterior_fn,
            bias_posterior_tensor_fn=bias_posterior_tensor_fn,
            bias_prior_fn=bias_prior_fn,
            bias_divergence_fn=bias_divergence_fn,
            **kwargs)

        self.output_padding = output_padding
        if self.output_padding is not None:
            self.output_padding = conv_utils.normalize_tuple(
                self.output_padding, 2, 'output_padding')
            for stride, out_pad in zip(self.strides, self.output_padding):
                if out_pad >= stride:
                    raise ValueError('Stride ' + str(self.strides) + ' must be '
                                     'greater than output padding ' +
                                     str(self.output_padding))

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input shape: ' +
                             str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined, Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        # If self.dtype is None, build weights using the default dtype.
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

        # Must have a posterior kernel
        self.kernel_posterior = self.kernel_posterior_fn(
            dtype, kernel_shape, 'kernel_posterior',
            self.trainable, self.add_variable)

        if self.kernel_prior_fn is None:
            self.kernel_prior = None
        else:
            self.kernel_prior = self.kernel_prior_fn(
                dtype, kernel_shape, 'kernel_prior',
                self.trainable, self.add_variable)
        self._built_kernel_divergence = False

        if self.bias_posterior_fn is None:
            self.bias_posterior = None
        else:
            self.bias_posterior = self.bias_posterior_fn(
                dtype, (self.filters,), 'bias_posterior',
                self.trainable, self.add_variable)

        if self.bias_prior_fn is None:
            self.bias_prior = None
        else:
            self.bias_prior = self.bias_prior_fn(
                dtype, (self.filters,), 'bias_prior',
                self.trainable, self.add_variable)
        self._built_bias_divergence = False

        self.built = True

    def call(self, inputs):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        self.output_shape_tensor = array_ops.stack(output_shape)
        outputs = self._apply_variational_kernel(inputs)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        outputs = self._apply_variational_bias(outputs)

        if self.activation is not None:
            outputs = self.activation(outputs)
        if not self._built_kernel_divergence:
            self._apply_divergence(self.kernel_divergence_fn,
                                   self.kernel_posterior,
                                   self.kernel_prior,
                                   self.kernel_posterior_tensor,
                                   name='divergence_kernel')
            self._built_kernel_divergence = True
        if not self._built_bias_divergence:
            self._apply_divergence(self.bias_divergence_fn,
                                   self.bias_posterior,
                                   self.bias_prior,
                                   self.bias_posterior_tensor,
                                   name='divergence_bias')
            self._built_bias_divergence = True
        return outputs

    def _apply_variational_kernel(self, inputs):
        self.kernel_posterior_tensor = self.kernel_posterior_tensor_fn(
            self.kernel_posterior)
        self.kernel_posterior_affine = None
        self.kernel_posterio_affine_tensor = None
        outputs = backend.conv2d_transpose(
            inputs,
            self.kernel_posterior_tensor,
            self.output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)
        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis, w_axis = 1, 2, 3
        else:
            c_axis, h_axis, w_axis = 3, 1, 2

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_output_length(
            output_shape[h_axis],
            kernel_h,
            padding=self.padding,
            output_padding=out_pad_h,
            stride=stride_h,
            dilation=self.dilation_rate[0])
        output_shape[w_axis] = conv_utils.deconv_output_length(
            output_shape[w_axis],
            kernel_w,
            padding=self.padding,
            output_padding=out_pad_w,
            stride=stride_w,
            dilation=self.dilation_rate[1])
        return tensor_shape.TensorShape(output_shape)

    def get_config(self):
        config = super(Conv2DTransposeReparameterization, self).get_config()
        config['output_padding'] = self.output_padding
        return config
