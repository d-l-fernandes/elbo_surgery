import tensorflow as tf
import tensorflow_probability as tfp

import aux
import base_model as bm
import mvg_dist.distributions as mvg_dist

ed = tfp.edward2
tfd = tfp.distributions


class VAEModel(bm.BaseModel):
    def __init__(self, config):
        super(VAEModel, self).__init__(config)

        self.t_y = None

        self.t_encoder_prod = None
        self.t_encoder_sum = None
        self.t_encoder = None

        self.t_z = None

        self.t_decoder = None

        self.t_prior = None

        self.t_full_reco = None
        self.t_avg_reco = None
        self.t_avg_kl = None
        self.t_avg_elbo_loss = None

        self.opt_trainer = None

        self.build_model()
        self.init_saver()

    def build_model(self):
        input_shape = [None] + self.config["state_size"]

        # Data
        self.t_y = tf.placeholder(shape=input_shape, dtype=self.dtype, name='Y')

        # Dimensions
        t_vae_q = tf.constant(self.config["vae_q"], dtype=tf.int32, name="Q_vae")

        self.t_encoder_prod, self.t_encoder_sum = aux.make_encoder(self.t_y, self.config["vae_q"],
                                                                   self.config["encoder_hidden_size"])

        if self.config["train_type"] == "product":
            self.t_z = self.t_encoder_prod.sample()
            self.t_encoder = self.t_encoder_prod
        else:
            self.t_z = self.t_encoder_sum.sample()
            self.t_encoder = self.t_encoder_sum

        if self.config["test_type"] == "product":
            self.t_encoder_test = self.t_encoder_prod
        else:
            self.t_encoder_test = self.t_encoder_sum

        self.t_decoder = aux.make_decoder(
            self.t_z,
            y_shape=tuple(input_shape[1:]),
            architecture_params=self.config["architecture_params"],
            architecture=self.config["architecture"],
            activation=self.config["activation"],
            output_distribution=self.config["output_distribution"]
        )

        self.t_prior = mvg_dist.MultivariateNormalLogDiag(tf.zeros([t_vae_q]), tf.zeros([t_vae_q]))

        # ELBO
        self.t_full_reco = self.t_decoder.log_prob(self.t_y)
        self.t_avg_reco = tf.reduce_mean(self.t_full_reco)
        self.t_avg_kl = tf.reduce_mean(tfd.kl_divergence(self.t_encoder, self.t_prior))
        self.t_avg_elbo_loss = \
            self.t_avg_reco - self.t_avg_kl - tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) / \
            tf.constant(self.config["num_data_points"], dtype=self.dtype)

        # Trainer
        self.opt_trainer = tf.contrib.opt.NadamOptimizer(
            learning_rate=self.config["learning_rate"]).minimize(-self.t_avg_elbo_loss)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
