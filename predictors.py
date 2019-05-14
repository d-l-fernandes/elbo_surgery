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
        # self._predict_from_y()
        self._predict_from_x()

    def _predict_from_y(self):

        self.get_label = self.config["label_latent_manifold"]
        # name of metric: [variable, if mean is to be used, if it goes in summary]
        self.metrics = {
            "elbo": [[], True, True],
            "reco": [[], True, True],
            "hist_reco": [[], False, True],
            "kl_prod": [[], True, True],
            "kl_sum": [[], True, True],
            "hist_kl_sum": [[], False, False],
            "mu_x": [[], False, False],
            "S_x": [[], False, False],
            "x_samples": [[], False, False],
            "elbo_test": [[], True, True],
            "reco_test": [[], True, True],
            "kl_prod_test": [[], True, True],
            "kl_sum_test": [[], True, True]
        }

        if self.get_label:
            self.metrics["labels"] = [[], False, False]

        if self.config["architecture"] == "bayes_dense":
            self._plot_weights()

        print("    Getting metrics...")
        self.batch_gen = self.data.select_batch_generator("testing_y")
        self.batch_gen_test = self.data.select_batch_generator("test_set")
        self.num_draws_per_batch = np.ceil(self.config["num_draws"] / self.config["num_iter_per_epoch"])
        loop = tqdm(range(self.config["num_iter_per_epoch"]), desc="Y Testing Epoch", ascii=True)

        for _ in loop:
            metrics_step = self._train_local_step()
            self.metrics = self.update_metrics_dict(self.metrics, metrics_step)

        self._plot_individual_kl()

        if self.config["vae_q"] <= 10:
            self._plot_latent_space()

        if self.config["plot_dimensions"] == 1:
            self.plot_1d_results(self.config["results_dir"],
                                 self.data.input_train,
                                 self.model.t_decoder,
                                 self.model.t_y,
                                 self.sess,
                                 self.config["batch_size"],
                                 self.config["num_iter_per_epoch"])

        print("    Getting Marginal KL...")
        marginal_kl_prod = self._calculate_marginal_kl(np.array(self.metrics["x_samples"][0]), "product")
        marginal_kl_sum = self._calculate_marginal_kl(np.array(self.metrics["x_samples"][0]), "sum")

        summaries_dict = self.create_summaries_dict(self.metrics)
        mutual_info_prod = summaries_dict["Metrics/kl_prod"] - marginal_kl_prod
        mutual_info_sum = summaries_dict["Metrics/kl_sum"] - marginal_kl_sum

        summaries_dict["Metrics/marginal_kl_prod"] = marginal_kl_prod
        summaries_dict["Metrics/marginal_kl_sum"] = marginal_kl_sum
        summaries_dict["Metrics/mutual_info_prod"] = mutual_info_prod
        summaries_dict["Metrics/mutual_info_sum"] = mutual_info_sum

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

        cost, reco, reco_full, kl_prod, kl_sum, kl_sum_full, mu_x, s_x, x_sample, reco_image = \
            self.sess.run((self.model.t_avg_elbo_loss,
                           self.model.t_avg_reco,
                           self.model.t_full_reco,
                           self.model.t_avg_kl_prod,
                           self.model.t_avg_kl_sum,
                           self.model.t_avg_kl_sum_full,
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
            "hist_kl_sum": np.array(kl_sum_full).T,
            "mu_x": mu_x,
            "x_samples": x_sample,
            "S_x": s_x
        }

        if self.get_label:
            metrics["labels"] = batch_label

        if self.get_label:
            batch_y, batch_y_original, batch_label = next(self.batch_gen_test)
        else:
            batch_y, batch_y_original = next(self.batch_gen_test)

        feed_dict = {self.model.t_y: batch_y}

        cost, reco, kl_prod, kl_sum = \
            self.sess.run((self.model.t_avg_elbo_loss,
                           self.model.t_avg_reco,
                           self.model.t_avg_kl_prod,
                           self.model.t_avg_kl_sum),
                          feed_dict)

        metrics["elbo_test"] = cost
        metrics["reco_test"] = reco
        metrics["kl_prod_test"] = kl_prod
        metrics["kl_sum_test"] = kl_sum

        return metrics

    def _plot_latent_space(self):
        image_name = os.path.join(self.config["results_dir"], f"xspace_")

        index = np.random.randint(0, self.config["num_data_points"], size=15000)
        x = np.array(self.metrics["mu_x"][0])[index]
        kls = np.array(self.metrics["hist_kl_sum"][0])[index]
        # kl_max = np.max(kls)
        kl_max = 8

        data = {f"x{i}": x[:, i] for i in range(self.config["vae_q"])}
        kls = {f"kl{i}": kls[:, i] for i in range(self.config["vae_q"])}
        data.update(kls)

        if 'labels' in self.metrics:
            labels = np.array(self.metrics["labels"][0])[index]
            data["color"] = labels
        else:
            labels = None

        data = pd.DataFrame.from_dict(data)
        brush = alt.selection(type='interval', resolve='global')
        latent_plots = None
        kl_plot = None

        for dim in range(self.config["vae_q"] // 2):
            if labels is None:
                scatter = alt.Chart().mark_circle().encode(
                    x=f'x{dim*2}',
                    y=f'x{dim*2+1}',
                    color=alt.condition(brush, alt.ColorValue('blue'), alt.ColorValue('gray')),
                    size=alt.value(5),
                    opacity=alt.value(0.5)
                ).add_selection(brush)
            else:
                scatter = alt.Chart().mark_circle().encode(
                    x=f'x{dim*2}',
                    y=f'x{dim*2+1}',
                    color=alt.condition(brush, alt.Color('color:O', scale=alt.Scale(scheme='category10')),
                                        alt.ColorValue('gray')),
                    size=alt.value(5),
                    opacity=alt.value(0.5)
                ).add_selection(brush)

            if latent_plots is None:
                latent_plots = scatter
            else:
                latent_plots = latent_plots | scatter

            hist = alt.Chart().mark_bar().encode(
                y=alt.Y(f'mean(kl{dim*2})', scale=alt.Scale(domain=[0, kl_max]))
            ).transform_filter(brush)
            hist = hist | alt.Chart().mark_bar().encode(
                y=alt.Y(f'mean(kl{dim*2+1})', scale=alt.Scale(domain=[0, kl_max]))
            ).transform_filter(brush)
            if kl_plot is None:
                kl_plot = hist
            else:
                kl_plot = kl_plot | hist

        final_plot = alt.vconcat(latent_plots, kl_plot, data=data)

        final_plot.save(f"{image_name}latent.html")

    # def _plot_latent_space(self, x, error_x, labels=None):
    #     image_name = os.path.join(self.config["results_dir"], f"xspace_")

    #     for dim in range(self.config["vae_q"] // 2):
    #         if labels is not None:
    #             data = {"x": x[:, dim*2],
    #                     "y": x[:, dim*2+1],
    #                     "s": error_x,
    #                     "color": labels}
    #             data = pd.DataFrame.from_dict(data)
    #             heat = alt.Chart(data).mark_rect().encode(
    #                 x=alt.X('x:Q', bin=alt.Bin(maxbins=60)),
    #                 y=alt.Y('y:Q', bin=alt.Bin(maxbins=60)),
    #                 color=alt.Color('average(s)', scale=alt.Scale(scheme="greys"))
    #             )
    #             scatter = alt.Chart(data).mark_circle().encode(
    #                 x='x:Q',
    #                 y='y:Q',
    #                 color=alt.Color('color:O', scale=alt.Scale(scheme="category10")),
    #                 size=alt.value(5),
    #                 opacity=alt.value(0.5)
    #             )
    #         else:
    #             data = {"x": x[:, dim*2],
    #                     "s": error_x,
    #                     "y": x[:, dim*2+1]}
    #             data = pd.DataFrame.from_dict(data)
    #             heat = alt.Chart(data).mark_rect().encode(
    #                 x=alt.X('x:Q', bin=alt.Bin(maxbins=60)),
    #                 y=alt.Y('y:Q', bin=alt.Bin(maxbins=60)),
    #                 color=alt.Color('average(s)', scale=alt.Scale(scheme="greys"))
    #             )
    #             scatter = alt.Chart(data).mark_circle().encode(
    #                 x='x:Q',
    #                 y='y:Q',
    #                 size=alt.value(5),
    #                 opacity=alt.value(0.5)
    #             )
    #         chart = alt.layer(heat, scatter).interactive()
    #         chart.save(f"{image_name}{dim*2+1}_{dim*2+2}.html")

    def _plot_individual_kl(self):
        kl_divergences = np.mean(self.metrics["hist_kl_sum"][0], axis=0)

        data = {"KL": kl_divergences,
                "dim": list(range(self.config["vae_q"]))
                }
        data = pd.DataFrame.from_dict(data)

        chart = alt.Chart(data).mark_bar().encode(
            x="dim:O",
            y="KL:Q"
        ).interactive()

        chart.save(f"{self.config['results_dir']}kl_divergences.html")

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

    def _get_encoder_prob(self, sample, test_type):
        batch_gen = self.data.select_batch_generator("testing_y")
        prob = 0
        for _ in range(self.config["num_iter_per_epoch"]):
            if self.get_label:
                batch_y, batch_y_original, batch_label = next(batch_gen)
            else:
                batch_y, batch_y_original = next(batch_gen)
            feed_dict = {self.model.t_y: batch_y}
            if test_type == "product":
                prob += self.sess.run(tf.reduce_sum(self.model.t_encoder_prod.prob(sample), axis=1),
                                      feed_dict=feed_dict)
            else:
                prob += self.sess.run(tf.reduce_mean(
                    [tf.reduce_sum(self.model.t_encoder_sum[i].prob(tf.expand_dims(sample[:, :, i], axis=-1)), axis=1)
                     for i in range(self.config["vae_q"])],
                    axis=0), feed_dict=feed_dict)
        return prob

    def _calculate_marginal_kl(self, x_samples, test_type):
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

        if test_type == "product":
            marginal_kl -= self.sess.run(tf.reduce_sum(self.model.t_prior_prod.log_prob(expanded_x_samples)[:, 0]),
                                         feed_dict)
        else:
            marginal_kl -= self.sess.run(
                tf.reduce_sum(
                    tf.reduce_mean(
                        [self.model.t_prior_sum.log_prob(tf.expand_dims(expanded_x_samples[:, :, i], axis=-1))[:, 0]
                         for i in range(self.config["vae_q"])],
                        axis=0)
                ),
                feed_dict)
        marginal_kl += np.sum(np.log(self._get_encoder_prob(expanded_x_samples, test_type)))

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

    def _predict_from_x(self):
        self.batch_gen = self.data.select_batch_generator("testing_x")

        for batch_mu_x in self.batch_gen:
            y_reco = self.sess.run(self.model.t_decoder.mean(),
                                   feed_dict={self.model.t_z: batch_mu_x})
            self.y_reco_list += y_reco.tolist()

        self._plot_x_grid()

    def _plot_x_grid(self):

        image_name = os.path.join(self.config["results_dir"], f"xdims")
        n_points = self.config["num_plot_x_points"]
        plt.clf()
        if self.config["vae_q"] == 1:
            if self.config["plot_dimensions"] == 3:
                f, axarr = plt.subplots(1, n_points, subplot_kw={"projection": '3d'})
            else:
                f, axarr = plt.subplots(1, n_points)
            for point in range(n_points):
                self.data.plot_data_point(self.y_reco_list[point], axarr[point])
                axarr[point].set_aspect("equal")
            plt.subplots_adjust(wspace=0, hspace=0)
            f.savefig(f"{image_name}_1.png", bbox_inches="tight", pad_inches=0, dpi=1600)
            plt.close(f)
        else:
            if self.config["vae_q"] % 2 == 1:
                if self.config["plot_dimensions"] == 3:
                    f, axarr = plt.subplots(1, n_points, subplot_kw={"projection": '3d'})
                else:
                    f, axarr = plt.subplots(1, n_points)
                for point in range(n_points):
                    self.data.plot_data_point(self.y_reco_list[self.data.total_n_points - n_points + point],
                                              axarr[point])
                    axarr[point].set_aspect("equal")
                plt.subplots_adjust(wspace=0, hspace=0)
                f.savefig(f"{image_name}_{self.config['gp_q']}.png", bbox_inches="tight", pad_inches=0, dpi=1600)
                plt.close(f)

            for dim in range(self.config["vae_q"] // 2):
                if self.config["plot_dimensions"] == 3:
                    f, axarr = plt.subplots(n_points, n_points, subplot_kw={"projection": '3d'})
                else:
                    f, axarr = plt.subplots(n_points, n_points)
                f.set_size_inches(10, 10)
                for j in range(n_points):
                    for i in range(n_points):
                        self.data.plot_data_point(self.y_reco_list[n_points**2 * dim + i + j*n_points],
                                                  axarr[n_points - (j+1), i])
                        axarr[n_points - (j+1), i].set_aspect("equal")
                f.subplots_adjust(wspace=0, hspace=0)
                f.savefig(f"{image_name}_{dim*2+1}_{dim*2+2}.png", bbox_inches="tight", pad_inches=0, dpi=1600)
                plt.close(f)
