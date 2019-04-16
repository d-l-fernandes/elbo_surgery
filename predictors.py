import os

import altair as alt
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from aux import jacobian
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

import base_model as bm


class VAEPredictor(bm.BasePredict):
    def __init__(self, sess, model, data, config, logger):
        super(VAEPredictor, self).__init__(sess, model, data, config, logger)

        self.batch_gen = None
        self.images_saved = 0

        self.y_reco_list = []

    def predict(self):
        print("Getting predictions...")
        self._predict_from_y()

    def _predict_from_y(self):

        self.get_label = self.config["label_latent_manifold"]
        # name of metric: [variable, if mean is to be used, if it goes in summary]
        self.metrics = {
            "elbo": [[], True, True],
            "reco": [[], True, True],
            "hist_reco": [[], False, True],
            "kl_prod": [[], True, True],
            "kl_sum": [[], True, True],
            "mu_x": [[], False, False],
            "S_x": [[], False, False],
            "x_samples": [[], False, False]
        }

        if self.get_label:
            self.metrics["labels"] = [[], False, False]

        if self.config["architecture"] == "bayes_dense":
            self._plot_weights()

        print("    Getting metrics...")
        self.batch_gen = self.data.select_batch_generator("testing_y")
        self.num_draws_per_batch = np.ceil(self.config["num_draws"] / self.config["num_iter_per_epoch"])
        loop = tqdm(range(self.config["num_iter_per_epoch"]), desc="Y Testing Epoch", ascii=True)

        for _ in loop:
            metrics_step = self._train_local_step()
            self.metrics = self.update_metrics_dict(self.metrics, metrics_step)

        if self.config["vae_q"] <= 10:
            if self.get_label:
                self._plot_latent_space(np.array(self.metrics["mu_x"][0]),
                                        np.array(self.metrics["S_x"][0]),
                                        np.array(self.metrics["labels"][0]))
            else:
                self._plot_latent_space(np.array(self.metrics["mu_x"][0]),
                                        np.array(self.metrics["S_x"][0]))

        if self.config["plot_dimensions"] == 1:
            self.plot_1d_results(self.config["results_dir"],
                                 self.data.input_train,
                                 self.model.t_decoder,
                                 self.model.t_y,
                                 self.sess,
                                 self.config["batch_size"],
                                 self.config["num_iter_per_epoch"])

        print("    Getting Marginal KL...")
        marginal_kl = self._calculate_marginal_kl(np.array(self.metrics["x_samples"][0]))

        summaries_dict = self.create_summaries_dict(self.metrics)
        if self.config["test_type"] == "product":
            mutual_info = summaries_dict["Metrics/kl_prod"] - marginal_kl
        else:
            mutual_info = summaries_dict["Metrics/kl_sum"] - marginal_kl

        summaries_dict["Metrics/marginal_kl"] = marginal_kl
        summaries_dict["Metrics/mutual_info"] = mutual_info

        self.logger.summarize(1, summaries_dict=summaries_dict, summarizer="test")

    def _train_local_step(self):
        """
        Training steps for local variables. Needed only for predict_from_y
        :return: several metrics + mu_x + saves a one data image (both original and reconstructed) as png
        """

        batch_label = None
        if self.get_label:
            batch_y, batch_y_original, batch_label = next(self.batch_gen)
        else:
            batch_y, batch_y_original = next(self.batch_gen)

        feed_dict = {self.model.t_y: batch_y}

        cost, reco, reco_full, kl_prod, kl_sum, mu_x, s_x, x_sample, reco_image = \
            self.sess.run((self.model.t_avg_elbo_loss,
                           self.model.t_avg_reco,
                           self.model.t_full_reco,
                           self.model.t_avg_kl_prod,
                           self.model.t_avg_kl_sum,
                           self.model.t_encoder_prod.mean(),
                           self.model.t_encoder_prod.stddev(),
                           self.model.t_encoder_prod.sample(self.num_draws_per_batch),
                           self.model.t_decoder.mean()),
                          feed_dict)

        if self.config["plot_dimensions"] != 1:
            self._plot_reconstructed_image(reco_image, batch_y_original)

        s_x = np.mean(s_x, axis=-1)
        metrics = {
            "elbo": cost,
            "reco": reco,
            "hist_reco": reco_full,
            "kl_prod": kl_prod,
            "kl_sum": kl_sum,
            "mu_x": mu_x,
            "x_samples": x_sample,
            "S_x": s_x
        }
        if self.get_label:
            metrics["labels"] = batch_label

        return metrics

    def _plot_latent_space(self, x, error_x, labels=None):
        image_name = os.path.join(self.config["results_dir"], f"xspace_")

        for dim in range(self.config["vae_q"] // 2):
            if labels is not None:
                data = {"x": x[:, dim*2],
                        "y": x[:, dim*2+1],
                        "s": error_x,
                        "color": labels}
                data = pd.DataFrame.from_dict(data)
                heat = alt.Chart(data).mark_rect().encode(
                    x=alt.X('x:Q', bin=alt.Bin(maxbins=60)),
                    y=alt.Y('y:Q', bin=alt.Bin(maxbins=60)),
                    color=alt.Color('average(s)', scale=alt.Scale(scheme="greys"))
                )
                scatter = alt.Chart(data).mark_circle().encode(
                    x='x:Q',
                    y='y:Q',
                    color=alt.Color('color:O', scale=alt.Scale(scheme="category10")),
                    size=alt.value(5),
                    opacity=alt.value(0.5)
                )
            else:
                data = {"x": x[:, dim*2],
                        "s": error_x,
                        "y": x[:, dim*2+1]}
                data = pd.DataFrame.from_dict(data)
                heat = alt.Chart(data).mark_rect().encode(
                    x=alt.X('x:Q', bin=alt.Bin(maxbins=60)),
                    y=alt.Y('y:Q', bin=alt.Bin(maxbins=60)),
                    color=alt.Color('average(s)', scale=alt.Scale(scheme="greys"))
                )
                scatter = alt.Chart(data).mark_circle().encode(
                    x='x:Q',
                    y='y:Q',
                    size=alt.value(5),
                    opacity=alt.value(0.5)
                )
            chart = alt.layer(heat, scatter).interactive()
            chart.save(f"{image_name}{dim*2+1}_{dim*2+2}.html")

    def _plot_reconstructed_image(self, reco, original):
        index_to_save = np.random.randint(self.config["batch_size"]-1)
        image_name = os.path.join(self.config["results_dir"], f"image{self.images_saved}")

        if self.config["plot_dimensions"] == 3:
            f, axarr = plt.subplots(1, 2, subplot_kw={"projection": '3d'})
        else:
            f, axarr = plt.subplots(1, 2)
        plt.subplots_adjust(wspace=0, hspace=0)
        self.data.plot_data_point(original[index_to_save], axarr[0])
        self.data.plot_data_point(reco[index_to_save], axarr[1])
        f.savefig(f"{image_name}.png", bbox_inches="tight", pad_inches=0)
        plt.close(f)

        self.images_saved += 1

    def _get_encoder_prob(self, sample):
        batch_gen = self.data.select_batch_generator("testing_y")
        prob = 0
        for _ in range(self.config["num_iter_per_epoch"]):
            if self.get_label:
                batch_y, batch_y_original, batch_label = next(batch_gen)
            else:
                batch_y, batch_y_original = next(batch_gen)
            feed_dict = {self.model.t_y: batch_y}
            if self.config["test_type"] == "product":
                prob += self.sess.run(tf.reduce_sum(self.model.t_encoder_prod.prob(sample), axis=1),
                                      feed_dict=feed_dict)
            else:
                prob += self.sess.run(tf.reduce_sum(
                    [tf.reduce_sum(self.model.t_encoder_sum[i].prob(tf.expand_dims(sample[:, :, i], axis=-1)), axis=1)
                     for i in range(self.config["vae_q"])],
                    axis=0
                ) / self.config["vae_q"], feed_dict=feed_dict)
        return prob

    def _calculate_marginal_kl(self, x_samples):
        """
        Calculates marginal KL (from ELBO surgery paper) for the test dataset

        :param x_samples: samples from encoder
        :return: marginal KL
        """

        marginal_kl = 0

        # Get num_draws_per_batch samples from each batch
        index1 = list(range(0, x_samples.shape[0]))
        index2 = np.random.randint(0, self.config["batch_size"], size=x_samples.shape[0])
        chosen_samples = x_samples[index1, index2]

        batch_gen = self.data.select_batch_generator("testing_y")
        if self.get_label:
            batch_y, batch_y_original, batch_label = next(batch_gen)
        else:
            batch_y, batch_y_original = next(batch_gen)
        feed_dict = {self.model.t_y: batch_y}
        expanded_x_samples = tf.cast(
            tf.tile(tf.expand_dims(chosen_samples, 1), [1, self.config["batch_size"], 1]),
            dtype=tf.float32)

        if self.config["test_type"] == "product":
            marginal_kl -= self.sess.run(tf.reduce_sum(self.model.t_prior_prod.log_prob(expanded_x_samples)[:, 0]),
                                         feed_dict)
        else:
            marginal_kl -= self.sess.run(tf.reduce_sum(
                [self.model.t_prior_sum.log_prob(tf.expand_dims(expanded_x_samples[:, :, i], axis=-1))[:, 0]
                 for i in range(self.config["vae_q"])]
            ) / self.config["vae_q"], feed_dict)
        marginal_kl += np.sum(np.log(self._get_encoder_prob(expanded_x_samples)))

        marginal_kl /= x_samples.shape[0]
        marginal_kl -= np.log(self.config["num_data_points"])

        return marginal_kl

    def _plot_weights(self):
        weights = [i for i in tf.trainable_variables("decoder") if "_loc" in i.name and "bias" not in i.name]

        chart = None
        for i, w in enumerate(weights):
            w_np = self.sess.run(w)
            w_np_max = int(np.ceil(np.max(np.abs(w_np))))
            x = np.tile(np.array([range(w_np.shape[1])]), (w_np.shape[0], 1))
            y = np.array([[i] * w_np.shape[1] for i in range(w_np.shape[0])])

            data = {"x": y.ravel(), "y": x.ravel(), "z": w_np.ravel()}
            data = pd.DataFrame.from_dict(data)
            new_chart = alt.Chart(data).mark_rect().encode(
                x='x:O',
                y='y:O',
                color=alt.Color("z:Q", scale=alt.Scale(scheme="redblue", domain=(-w_np_max, w_np_max)))
            )
            if chart is None:
                chart = new_chart
            else:
                chart = chart | new_chart

        image_name = os.path.join(self.config["results_dir"], "decoder_weights")
        chart.save(f"{image_name}.html")
