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
        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()

        self.input_train = self.binarize(x_train)
        self.input_test = self.binarize(x_test)

        self.input_train_non_bin = x_train
        self.input_test_non_bin = x_test

    def select_batch_generator(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train[idx]
        elif phase == "testing_y":
            for i in range(self.num_batches):
                yield self.input_train[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train_non_bin[i * self.b_size:(i+1) * self.b_size], \
                      self.y_train[i * self.b_size:(i+1) * self.b_size]
        elif phase == "test_set":
            while True:
                num_batches = self.input_test.shape[0] // self.b_size
                for i in range(num_batches):
                    yield self.input_test[i * self.b_size:(i+1) * self.b_size], \
                          self.input_test_non_bin[i * self.b_size:(i+1) * self.b_size], \
                          self.y_test[i * self.b_size:(i+1) * self.b_size]
        elif phase == "testing_x":
            num_iter, final_grid = self.testing_x_generator()
            for i in range(num_iter):
                yield final_grid[i * self.b_size:(i+1) * self.b_size]

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

    def select_batch_generator(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train[idx]
        elif phase == "testing_y":
            for i in range(self.num_batches):
                yield self.input_train[i * self.b_size:(i+1) * self.b_size], \
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

    def select_batch_generator(self, phase):
        if phase == "training":
            while True:
                idx = np.random.choice(self.config["num_data_points"], self.b_size)
                yield self.input_train[idx]
        elif phase == "testing_y":
            for i in range(self.num_batches):
                yield self.input_train[i * self.b_size:(i+1) * self.b_size], \
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


class DataGeneratorCelebA(bm.BaseDataGenerator):
    def __init__(self, config):
        super(DataGeneratorCelebA, self).__init__(config)

        ds_train = tf.data.TFRecordDataset("/home/dfernandes/Data/celebA/celebAWithAttr-train.tfrecords").map(
            self.parse_and_decode_example)

        # ds_train = ds_train.shuffle(buffer_size=self.config["num_data_points"])
        ds_train = ds_train.repeat()
        ds_train = ds_train.batch(self.config["batch_size"])
        self.input_train = ds_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.input_train_iter = self.input_train.make_one_shot_iterator().get_next()
        self.input_train_non_bin = self.input_train
        self.input_train_non_bin_iter = self.input_train_non_bin.make_one_shot_iterator().get_next()

        ds_test = tf.data.TFRecordDataset("/home/dfernandes/Data/celebA/celebAWithAttr-test.tfrecords").map(
            self.parse_and_decode_example)

        # ds_test = ds_test.shuffle(buffer_size=self.config["num_data_points"])
        ds_test = ds_test.batch(self.config["batch_size"])
        self.input_test = ds_test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.input_test_iter = self.input_test.make_one_shot_iterator().get_next()
        self.input_test_non_bin = self.input_test
        self.input_test_non_bin_iter = self.input_test_non_bin.make_one_shot_iterator().get_next()

        self.sess = tf.Session()

    @staticmethod
    def decode_and_reshape_image_feature(feature, shape, in_dtype=tf.uint8, out_dtype=tf.float32):
        # Decode the image
        image = tf.decode_raw(feature, in_dtype)
        image = tf.cast(image, out_dtype)

        # Vector to NxNxC
        if shape is not None:
            image = tf.reshape(image, shape)

        return image

    def parse_and_decode_example(self, serialized_example):
        # Retrieve the features of interest based on keys
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([1], tf.string),
                'attr': tf.FixedLenFeature([40], tf.int64),
                'id': tf.VarLenFeature(tf.string)
            })

        features['image'] = self.minmax(
            self.resize_img(tf.image.rgb_to_grayscale(
            self.decode_and_reshape_image_feature(features['image'], shape=(218, 178, 3)))), 255., 0.)

        features['id'] = features['id'].values

        # Remove extra dims, i.e id = [['name0'],...,['name16']] to ['name0',...,'name16']
        features['id'] = tf.squeeze(features['id'])
        features['id'].set_shape(())

        return tf.squeeze(features['image'])

    @staticmethod
    def resize_img(img, output_img_size=(64, 64), crop_box=(70, 35, 108, 108),
                   resize_method=tf.image.ResizeMethod.BILINEAR):
        img = tf.image.crop_to_bounding_box(img, crop_box[0], crop_box[1], crop_box[2], crop_box[3])

        if not np.array_equal(crop_box[2:], output_img_size):
            img = tf.image.resize_images(img, output_img_size, method=resize_method)

        return img

    def select_batch_generator(self, phase):
        if phase == "training":
            while True:
                train, train_non_bin = self.sess.run((self.input_train_iter, self.input_train_non_bin_iter))
                yield train
        elif phase == "testing_y":
            for i in range(self.num_batches):
                yield self.input_train[i * self.b_size:(i+1) * self.b_size], \
                      self.input_train_non_bin[i * self.b_size:(i+1) * self.b_size]
        elif phase == "test_set":
            while True:
                num_batches = self.input_test.shape[0] // self.b_size
                for i in range(num_batches):
                    yield self.input_test[i * self.b_size:(i+1) * self.b_size], \
                          self.input_test_non_bin[i * self.b_size:(i+1) * self.b_size]
        elif phase == "testing_x":
            num_iter, final_grid = self.testing_x_generator()
            for i in range(num_iter):
                yield final_grid[i * self.b_size:(i+1) * self.b_size]

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
