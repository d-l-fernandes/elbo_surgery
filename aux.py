import numpy as np

import tensorflow as tf
from tensorflow.python.keras import layers

import tensorflow_probability as tfp


import bayes_layers
import mvg_dist.distributions as mvg_dist

tfd = tfp.distributions
ed = tfp.edward2


# Concrete dropout stuff
class ConcreteDropout(layers.Wrapper):
    """This wrapper allows to learn the dropout probability
        for any given input layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$
             (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and
             N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eucledian
            loss.

    # Warning
        You must import the actual layer class from tf layers,
         else this will not work.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = (np.log(init_min) - np.log(1. - init_min))
        self.init_max = (np.log(init_max) - np.log(1. - init_max))

    def build(self, input_shape=None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if hasattr(self.layer, 'built') and not self.layer.built:
            self.layer.build(input_shape)

        # initialise p
        self.p_logit = self.add_variable(name='p_logit',
                                         shape=(1,),
                                         initializer=tf.keras.initializers.random_uniform(self.init_min, self.init_max),
                                         dtype=tf.float32,
                                         trainable=True)

        self.p = tf.nn.sigmoid(self.p_logit[0])
        tf.add_to_collection("LAYER_P", self.p)

        # initialise regulariser / prior KL term
        input_dim = int(np.prod(input_shape[1:]))

        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(
            weight)) / (1. - self.p)
        dropout_regularizer = self.p * tf.log(self.p)
        dropout_regularizer += (1. - self.p) * tf.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)
        # Add the regularisation loss to collection.
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        """
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        """
        eps = 1e-7
        temp = 0.1

        unif_noise = tf.random_uniform(shape=tf.shape(x))
        drop_prob = (
            tf.log(self.p + eps)
            - tf.log(1. - self.p + eps)
            + tf.log(unif_noise + eps)
            - tf.log(1. - unif_noise + eps)
        )
        drop_prob = tf.nn.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        return self.layer.call(self.concrete_dropout(inputs))


# TF stuff
def create_real_variable(initial_value, name, dtype=tf.float32):
    return tf.Variable(initial_value=initial_value,
                       name=name,
                       dtype=dtype)


def jacobian(y, x, y_dims):
    j = []
    for i in range(y_dims):
        j.append(tf.gradients(y[:, i:i+1], x)[0])

    return tf.transpose(tf.convert_to_tensor(j), [1, 0, 2])


# VAE stuff
def make_encoder(y, latent_d, hidden_size=(500, 500)):
    net = make_dense(y, latent_d * 2, {"hidden_size": hidden_size}, concrete=False)
    dist_prod = mvg_dist.MultivariateNormalLogDiag(
        loc=net[..., :latent_d],
        log_covariance_diag=net[..., latent_d:],
        name="encoder"
    )

    dist_sum = [
        mvg_dist.MultivariateNormalLogDiag(
            loc=net[..., i:i+1],
            log_covariance_diag=net[...,latent_d+i:latent_d+i+1]
        )
        for i in range(latent_d)
    ]
    return dist_prod, dist_sum


def make_decoder(x_code,
                 y_shape,
                 architecture_params,
                 architecture="bayes_dense",
                 activation="softplus",
                 output_distribution="bernoulli"):

    possible_architectures = ["bayes_dense", "bayes_conv", "regular", "concrete"]
    possible_output_dists = ["bernoulli", "gaussian"]
    possible_activations = ["softplus", "sigmoid", "relu", "tanh"]

    assert architecture in possible_architectures
    assert output_distribution in possible_output_dists
    assert activation in possible_activations

    if output_distribution == "gaussian":
        output_shape = np.prod(y_shape) * 2
    else:
        output_shape = np.prod(y_shape)

    if activation == "softplus":
        activ_function = tf.nn.softplus
    elif activation == "relu":
        activ_function = tf.nn.relu
    elif activation == "tanh":
        activ_function = tf.nn.tanh
    else:
        activ_function = tf.nn.sigmoid

    if architecture == "bayes_dense":
        net = make_bayes_dense(x_code, output_shape, architecture_params, activ_function)
    elif architecture == "bayes_conv":
        net = make_bayes_conv(x_code, y_shape, architecture_params, activ_function)
    elif architecture == "concrete":
        net = make_dense(x_code, output_shape, architecture_params, activ_function, concrete=True)
    else:
        net = make_dense(x_code, output_shape, architecture_params, activ_function, concrete=False)

    dimensions = len(y_shape)
    if output_distribution == "gaussian":
        means = tf.reshape(
            net[..., :np.prod(y_shape)], tf.concat([[-1], y_shape], axis=0),
            name="means"
        )
        log_variance = tf.reshape(net[..., np.prod(y_shape):], tf.concat([[-1], y_shape], axis=0), name="log_variance")
        return mvg_dist.MultivariateNormalLogDiag(loc=means,
                                                  log_covariance_diag=log_variance)
    else:
        logits = tf.reshape(
            net, tf.concat([[-1], y_shape], axis=0)
        )
        return tfd.Independent(tfd.Bernoulli(logits=logits, dtype=tf.float64), dimensions)


def make_dense(x, out_size, architecture_params, activation=tf.nn.sigmoid, concrete=True):
    model_layers = [tf.keras.layers.Flatten()]
    for h in architecture_params["hidden_size"]:
        if concrete:
            model_layers.append(ConcreteDropout(tf.keras.layers.Dense(h, activation=activation), trainable=True))
        else:
            model_layers.append(tf.keras.layers.Dense(h, activation=activation))
    if concrete:
        model_layers.append(ConcreteDropout(tf.keras.layers.Dense(out_size), trainable=True))
    else:
        model_layers.append(tf.keras.layers.Dense(out_size))
    model = tf.keras.Sequential(model_layers)
    net = model(x, training=True)
    return net


def make_bayes_dense(x, out_size, architecture_params, activation=tf.nn.softplus):
    model_layers = [tf.keras.layers.Flatten()]
    for h in architecture_params["hidden_size"]:
        model_layers.append(tfp.layers.DenseReparameterization(
            h,
            activation=activation,
            kernel_posterior_fn=bayes_layers.multivariate_normal_fn(),
            kernel_prior_fn=bayes_layers.multivariate_normal_gamma_precision_fn()))
    model_layers.append(tfp.layers.DenseReparameterization(
        np.array(out_size, dtype=np.int32),
        kernel_posterior_fn=bayes_layers.multivariate_normal_fn(),
        kernel_prior_fn=bayes_layers.multivariate_normal_gamma_precision_fn()))
    model = tf.keras.Sequential(model_layers)
    net = model(x)
    regularizer = tf.reduce_sum(model.losses)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return net


def make_bayes_conv(x, out_size, architecture_params, activation=tf.nn.softplus):
    kernel_size = architecture_params["kernel_size"]
    strides = architecture_params["strides"]
    padding = architecture_params["padding"]
    filters = architecture_params["filters"]

    input_shape = x.shape.as_list()
    if len(input_shape) == 2:
        net = tf.matrix_diag(x)
        net = tf.expand_dims(net, axis=-1)
    elif len(input_shape) == 3:
        net = tf.expand_dims(x, axis=-1)
    elif len(input_shape) == 4:
        net = x
    else:
        raise RuntimeError(f'Input as invalid number of dimensions ({len(input_shape)}).' +
                           ' Valid numbers are 2, 3 and 4')
    model_layers = []

    for i, (s, f) in enumerate(zip(strides, filters)):
        if i == len(filters) - 1:
            break
        model_layers.append(bayes_layers.Conv2DTransposeReparameterization(
            f, kernel_size, s, padding, activation=activation,
            kernel_posterior_fn=bayes_layers.multivariate_normal_fn(),
            kernel_prior_fn=bayes_layers.multivariate_normal_gamma_precision_fn()))

    model_layers.append(bayes_layers.Conv2DTransposeReparameterization(
        filters[-1], kernel_size, strides[-1], padding,
        kernel_posterior_fn=bayes_layers.multivariate_normal_fn(),
        kernel_prior_fn=bayes_layers.multivariate_normal_gamma_precision_fn()))

    model = tf.keras.Sequential(model_layers)

    net = model(net)
    output_shape = net.shape.as_list()
    if output_shape[1] < out_size[0]:
        h_pad_up = (out_size[0] - output_shape[1]) // 2
        h_pad_down = out_size[0] - output_shape[1] - h_pad_up
        net = tf.pad(net, [[0, 0], [h_pad_up, h_pad_down], [0, 0], [0, 0]])
    else:
        net = net[:, :out_size[0], :, :]

    if output_shape[2] < out_size[1]:
        w_pad_up = (out_size[0] - output_shape[2]) // 2
        w_pad_down = out_size[0] - output_shape[2] - w_pad_up
        net = tf.pad(net, [[0, 0], [0, 0], [w_pad_up, w_pad_down], [0, 0]])
    else:
        net = net[:, :, :out_size[1], :]

    regularizer = tf.reduce_sum(model.losses)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularizer)

    return net
