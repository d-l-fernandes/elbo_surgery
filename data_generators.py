import tensorflow as tf
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split

import os

import warnings
import c3d
import base_model as bm


class DataGeneratorMNIST(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorMNIST, self).__init__(config)
        # load data here
        mnist = tf.keras.datasets.mnist
        (x_train, self.y_train), (x_test, y_test) = mnist.load_data()

        self.input_train = self.binarize(x_train)
        self.input_test = self.binarize(x_test)

        self.input_train_non_bin = x_train

        self.input_train_pca = self.pca_transform(self.input_train, self.config["gp_q"])

    def select_batch_generator(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train[idx], self.input_train_pca[idx]
        elif phase == "testing_y":
            for i in range(self.num_batches):
                yield self.input_train[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train_pca[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train_non_bin[i * self.b_size:(i+1) * self.b_size], \
                      self.y_train[i * self.b_size:(i+1) * self.b_size]

    def plot_data_point(self, data, axis):
        if type(data) is list:
            data = np.array(data)

        if len(data.shape) == 1:
            width = np.int(np.sqrt(data.shape[0]))
            data = data.reshape(width, width)
        axis.imshow(data, cmap="gray")
        axis.axis("off")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)


class DataGeneratorMocap(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorMocap, self).__init__(config)
        self.path = "/home/dfernandes/Data/mocap/data.npy"

        # load data here
        x_train = np.load(self.path)

        self.input_train = self.standardize(x_train)
        self.input_train_pca = self.pca_transform(self.input_train, self.config["gp_q"])

    def select_batch_generator(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train[idx], self.input_train_pca[idx]
        elif phase == "testing_y":
            for i in range(self.num_batches):
                yield self.input_train[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train_pca[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train[i * self.b_size:(i+1) * self.b_size]

    def plot_data_point(self, data, axis):
        x_coord = []
        y_coord = []
        z_coord = []

        if type(data) is list:
            data = np.array(data)

        data = self.reverse_standardize(data)
        for i in range(0, data.shape[0] - 2, 3):
            x_coord.append(data[i])
            y_coord.append(data[i+1])
            z_coord.append(data[i+2])

        axis.scatter(x_coord, y_coord, z_coord, marker="1", s=5, c="k")
        axis.axis("off")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)


class DataGeneratorCMUWalk(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorCMUWalk, self).__init__(config)

        self.path = "/home/dfernandes/Data/CMU_walk/"
        files = os.listdir(self.path)

        # load data here
        x_train = []
        for f in files:
            with open(f'{self.path}{f}', 'rb') as handle:
                reader = c3d.Reader(handle)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    for frame in reader.read_frames():
                        x_train.append([])
                        for i, body_part in enumerate(frame[1]):
                            if i == 41:  # some of the frames have 42 data_points...
                                break
                            x_train[-1] += body_part[:3].tolist()

        x_train = np.array(x_train)
        self.input_train = self.standardize(x_train)
        # self.input_train = self.minmax(x_train)
        self.input_train_pca = self.pca_transform(self.input_train, self.config["gp_q"])

    def select_batch_generator(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train[idx], self.input_train_pca[idx]
        elif phase == "testing_y":
            for i in range(self.num_batches):
                yield self.input_train[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train_pca[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train[i * self.b_size:(i+1) * self.b_size]

    def plot_data_point(self, data, axis):
        x_coord = []
        y_coord = []
        z_coord = []

        if type(data) is list:
            data = np.array(data)

        data = self.reverse_standardize(data)
        # data = self.reverse_minmax(data)
        for i in range(0, data.shape[0] - 2, 3):
            x_coord.append(data[i])
            y_coord.append(data[i+1])
            z_coord.append(data[i+2])

        axis.scatter(x_coord, y_coord, z_coord, marker="1", s=5, c="k")
        axis.axis("off")
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)


class DataGeneratorMixtureGaussians(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorMixtureGaussians, self).__init__(config)
        # load data here
        min_x = -10
        max_x = 10
        train_share = 0.8
        num_points = np.ceil(self.config["num_data_points"] / 0.8).astype(int)

        mu_list = [-9, -6, -3, 0, 3, 6, 9]
        normal_scale = 3

        slope = 0.1
        origin = 3

        x = np.linspace(min_x, max_x, num_points)
        y = slope * x + origin

        for mu in mu_list:
            y += normal_scale * stats.norm.pdf(x, mu)

        if self.config["gp_q"] > 2:
            random_dims = np.random.normal(size=(num_points, self.config["gp_q"]-2))
            y = np.hstack((np.expand_dims(y, axis=-1), random_dims))
        else:
            y = np.expand_dims(y, axis=-1)

        y = np.concatenate((np.expand_dims(x, axis=-1), y), axis=1)
        self.input_train, self.input_test = train_test_split(y, train_size=train_share)

        self.input_train_non_bin = self.input_train

        # self.input_train_pca = self.pca_transform(self.input_train, self.config["gp_q"])
        self.input_train_pca = self.input_train

    def select_batch_generator(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train[idx], self.input_train_pca[idx]
        elif phase == "testing_y":
            for i in range(self.num_batches):
                yield self.input_train[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train_pca[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train_non_bin[i * self.b_size:(i+1) * self.b_size]
