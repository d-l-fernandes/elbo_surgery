import os

import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_bvp
from sklearn.decomposition import PCA
import tensorflow as tf


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code:  0:sucess -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print(f"Creating directories error: {err}")
        exit(-1)


class Logger:
    def __init__(self, sess, config, predict):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}

        if predict == 0:
            self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], "train"),
                                                              self.sess.graph)
        elif predict == 1:
            self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], "train"),
                                                              self.sess.graph)
            self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], "test"))
        else:
            self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config["summary_dir"], "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """

        if summarizer == "train":
            summary_writer = self.train_summary_writer
        else:
            summary_writer = self.test_summary_writer

        with tf.variable_scope(scope):
            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder("float64", value.shape, name=tag)
                        else:
                            self.summary_placeholders[tag] = tf.placeholder("float64",
                                                                            [None] + list(value.shape[1:]), name=tag)
                        if len(value.shape) <= 1:
                            if "hist" in tag:
                                self.summary_ops[tag] = tf.summary.histogram(tag, self.summary_placeholders[tag])
                            else:
                                self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)

                summary_writer.flush()


class BaseModel:
    def __init__(self, config):
        self.config = config

        self.increment_cur_epoch_tensor = None
        self.cur_epoch_tensor = None
        self.global_step_tensor = None

        self.metric = -np.inf

        tf.reset_default_graph()

        # init the epoch counter
        self.init_cur_epoch()
        self.dtype = tf.float32
        self.saver = None

    # save function that saves the checkpoint in the path defined in the config file
    # only saves if the current model is better than the best
    def save(self, sess, cur_metric):
        if cur_metric > self.metric:
            self.metric = cur_metric
            print("Saving model...")
            self.saver.save(sess, self.config["checkpoint_dir"], self.cur_epoch_tensor)
            print("Model saved")
        else:
            print("Not saved, as the metric is worse")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config["checkpoint_dir"])
        if latest_checkpoint:
            print(f"Loading model checkpoint {latest_checkpoint} ...\n")
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")

    # just initialize a tensorflow variable to use as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope("cur_epoch"):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name="cur_epoch")
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor,
                                                        self.cur_epoch_tensor + tf.constant(1))

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config["max_to_keep"])
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError


class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.max_epoch_diff = 20  # The training will stop if the objective will not improve after this many epochs
        self.cur_epoch_diff = 0
        self.metric = -np.inf  # Current best value of objective
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config["num_epochs"], 1):
            objective = self.train_epoch(cur_epoch)

            if objective > self.metric:
                self.metric = objective
                self.cur_epoch_diff = 0
            else:
                self.cur_epoch_diff += 1

            self.sess.run(self.model.increment_cur_epoch_tensor)

            if self.cur_epoch_diff == self.max_epoch_diff:
                print("Training stopped as it was not improving")
                break

    def train_epoch(self, cur_epoch):
        """
        implement the logic of epoch:
        - loop over the number of iterations in the config and call the train step
        - add any summaries you want using the summary

        returns objective
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metric you need to summarize
        """
        raise NotImplementedError

    @staticmethod
    def update_metrics_dict(metrics_epoch, metrics_step):
        for metric in metrics_step.keys():
            if metrics_epoch[metric][1]:
                metrics_epoch[metric][0].append(metrics_step[metric])
            else:
                metrics_epoch[metric][0] += metrics_step[metric].tolist()

        return metrics_epoch

    @staticmethod
    def create_summaries_dict(metrics_epoch):
        summaries_dict = {}
        for metric in metrics_epoch.keys():
            if metrics_epoch[metric][1]:
                summaries_dict[f"Metrics/{metric}"] = np.mean(metrics_epoch[metric][0])
            else:
                summaries_dict[f"Metrics/{metric}"] = np.array(metrics_epoch[metric][0])

        return summaries_dict


class BasePredict:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data

    def predict(self):
        """
        Base class to do predictions
        :return: None, saves predictions to file
        """
        raise NotImplementedError

    @staticmethod
    def plot_1d_results(results_dir, y_data, model_out_dist, model_in, sess, batch_size, iters, num_samples=500):
        y_model = None
        y_model_mean = None
        y_model_std = None
        for i in range(iters):
            batch = y_data[i * batch_size:(i + 1) * batch_size]
            batch_model, batch_model_mean, batch_model_std = \
                sess.run((model_out_dist.sample(num_samples),
                          model_out_dist.mean(),
                          model_out_dist.stddev()),
                         feed_dict={model_in: batch})
            if y_model_mean is None:
                y_model = batch_model
                y_model_mean = batch_model_mean
                y_model_std = batch_model_std
            else:
                y_model = np.concatenate((y_model, batch_model), axis=1)
                y_model_mean = np.concatenate((y_model_mean, batch_model_mean), axis=0)
                y_model_std = np.concatenate((y_model_std, batch_model_std), axis=0)

        y_stoch_mean = np.mean(y_model, axis=0)
        y_stoch_std = np.std(y_model, axis=0)

        data = {"x": y_data[:, 0],
                "y_data": y_data[:, 1],
                "y_model_mean": y_model_mean[:, 1],
                "y_model_std_up": y_model_mean[:, 1] + y_model_std[:, 1],
                "y_model_std_down": y_model_mean[:, 1] - y_model_std[:, 1],
                "y_stoch_mean": y_stoch_mean[:, 1],
                "y_stoch_std_up": y_stoch_mean[:, 1] + y_stoch_std[:, 1],
                "y_stoch_std_down": y_stoch_mean[:, 1] - y_stoch_std[:, 1],
                }
        data = pd.DataFrame.from_dict(data)
        base = alt.Chart(data)
        data_points = base.mark_circle().encode(
            x="x:Q",
            y="y_data:Q",
            color=alt.value('black'),
            size=alt.value(5),
        )
        model_mean = base.mark_line().encode(
            x="x:Q",
            y="y_model_mean:Q",
            color=alt.value("blue")
        )
        model_std = base.mark_area(opacity=0.3).encode(
            x="x:Q",
            y=alt.Y('y_model_std_up', title=""),
            y2='y_model_std_down',
            color=alt.value("blue")
        )
        stoch_mean = base.mark_line().encode(
            x="x:Q",
            y="y_stoch_mean:Q",
            color=alt.value("red")
        )
        stoch_std = base.mark_area(opacity=0.3).encode(
            x="x:Q",
            y=alt.Y('y_stoch_std_up', title=""),
            y2='y_stoch_std_down',
            color=alt.value("red")
        )
        chart = alt.layer(model_std, model_mean, stoch_std, stoch_mean, data_points).interactive()
        chart.save(os.path.join(results_dir, "data.html"))

    @staticmethod
    def update_metrics_dict(metrics_epoch, metrics_step):
        for metric in metrics_step.keys():
            if metrics_epoch[metric][1]:
                metrics_epoch[metric][0].append(metrics_step[metric])
            else:
                metrics_epoch[metric][0] += metrics_step[metric].tolist()

        return metrics_epoch

    @staticmethod
    def create_summaries_dict(metrics_epoch):
        summaries_dict = {}
        for metric in metrics_epoch.keys():
            if not metrics_epoch[metric][2]:
                continue
            if metrics_epoch[metric][1]:
                summaries_dict[f"Metrics/{metric}"] = np.mean(metrics_epoch[metric][0])
            else:
                summaries_dict[f"Metrics/{metric}"] = np.array(metrics_epoch[metric][0])

        return summaries_dict

    @staticmethod
    def get_geodesic(fun, bc, x, y):
        solution = solve_bvp(fun, bc, x, y, max_nodes=1500)

        return solution

    @staticmethod
    def create_initial_mesh(ya, yb, initial_nodes=100):
        x = np.linspace(0, 1, initial_nodes)
        y = None
        for d in range(ya.shape[0]):
            yd = np.zeros(initial_nodes)
            yd[0] = ya[d]
            yd[-1] = yb[d]
            if y is None:
                y = yd
            else:
                y = np.vstack((y, yd))

        y_der = np.zeros((ya.shape[0], initial_nodes))
        y = np.vstack((y, y_der))
        return x, y


class BaseDataGenerator:
    def __init__(self, config):
        self.config = config
        self.total_n_points = 0
        self.num_batches = self.config["num_iter_per_epoch"]
        self.b_size = self.config["batch_size"]

        self.mean = None
        self.stddev = None

        self.min = None
        self.max = None

    def standardize(self, data):
        self.mean = np.mean(data, axis=0)
        self.stddev = np.std(data, axis=0)

        norm_data = (data - self.mean) / self.stddev

        return norm_data

    def reverse_standardize(self, data):
        return data * self.stddev + self.mean

    def minmax(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)

        minmax_data = (data - self.min) / (self.max - self.min)

        return minmax_data

    def reverse_minmax(self, data):
        return data * (self.max - self.min) + self.min

    @staticmethod
    def pca_transform(data, pca_dims, pca_fit=None):
        data_flat = np.reshape(data, [data.shape[0], np.prod(data.shape[1:])])
        if pca_fit is None:
            pca_fit = PCA(n_components=pca_dims)
            pca_fit.fit(data_flat)

        return pca_fit.transform(data_flat)

    @staticmethod
    def binarize(x: np.ndarray, boundary: float = 0.5, max_value: float = 255.0) -> np.ndarray:
        """Binarizes x' = x / max_value
        1 if x' > boundary
        0 if x' <= boundary
        :rtype: np.ndarray"""

        x = x / max_value

        return np.where(x > boundary, 1, 0)

    def select_batch_generator(self, phase):
        pass

    def plot_data_point(self, data, axis):
        pass
