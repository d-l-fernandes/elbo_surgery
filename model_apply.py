import argparse
import json
from multiprocessing import Pool
import os
import shutil
import warnings

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import base_model as bm
import config_dicts as cd
import data_generators as dg
import models
import predictors
import trainers


def run(config, model, trainer, predictor, data, args):
    model_instance = model(config)

    # Restrict amount of GPU memory to only what is needed
    gpu_options = tf.ConfigProto()
    gpu_options.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_options)

    if args.debug:
        warnings.filterwarnings("ignore")
        tf.logging.set_verbosity(tf.logging.ERROR)
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    logger = bm.Logger(sess, config, args.predict)

    if args.predict == 0:
        trainer_instance = trainer(sess, model_instance, data, config, logger)

        if args.restore:
            model_instance.load(sess)
            sess.run(model_instance.increment_cur_epoch_tensor)

        trainer_instance.train()
    elif args.predict == 1:
        trainer_instance = trainer(sess, model_instance, data, config, logger)
        predictor_instance = predictor(sess, model_instance, data, config, logger)

        if args.restore:
            model_instance.load(sess)
            sess.run(model_instance.increment_cur_epoch_tensor)

        trainer_instance.train()
        # Restore best model
        model_instance.load(sess)
        predictor_instance.predict()
    else:
        predictor_instance = predictor(sess, model_instance, data, config, logger)
        model_instance.load(sess)
        predictor_instance.predict()

    sess.close()


def main():

    parser = argparse.ArgumentParser(description="Trains VAE-GPLVM model")
    parser.add_argument("-d", "--debug", help="start TF debug", action="store_true")
    parser.add_argument("-p", "--predict",
                        help="run predictions (0 - only train, 1 - train and predict, 2 - only predict)",
                        action="count", default=0)
    parser.add_argument("-m", "--multi", help="use subprocess to release GPU memory", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", "--restore", help="restore checkpoint", action="store_true")
    group.add_argument("-e", "--erase", help="erase summary dirs before training", action="store_true")
    args = parser.parse_args()

    vae_q_list = [2, 4, 10, 20]
    vae_q_list = [10]

    # dataset = "mnist"
    # dataset = "cmu_walk"
    # dataset = "mocap"
    dataset = "celebA"

    model_name = "vae"

    # Decoder architecture
    # train_type = "product"
    train_type = "sum"

    parent_folder = f"results/{model_name}/{train_type}/{dataset}"

    # architecture = "bayes_dense"
    # architecture = "bayes_conv"
    architecture = "concrete"
    # architecture = "regular"

    activation = "softplus"
    # activation = "sigmoid"
    # activation = "tanh"
    # activation = "relu"

    if architecture == "bayes_dense":
        # Bayes hidden_layers
        architecture_params = {
            "hidden_size": (20, 20)
        }
    elif architecture == "bayes_conv":
        architecture_params = {
            "filters": (32, 64, 32, 1),
            "kernel_size": 4,
            "strides": (1, 1, 1, 1),
            "padding": "valid"
        }
    else:
        # Normal NN hidden layers
        architecture_params = {
            "hidden_size": (300, 300)
        }

    parent_folder += f"/{architecture}/{activation}"

    encoder_hidden_size = (300, 300)

    model = models.VAEModel
    trainer = trainers.VAETrainer
    predictor = predictors.VAEPredictor

    for vae_q in vae_q_list:
        config = cd.config_dict(parent_folder, vae_q, dataset)

        config["architecture"] = architecture
        config["activation"] = activation
        config["architecture_params"] = architecture_params
        config["encoder_hidden_size"] = encoder_hidden_size
        config["train_type"] = train_type

        # Create the experiment dirs
        if args.erase:
            if os.path.exists(config["summary_dir"]):
                shutil.rmtree(config["summary_dir"])
                shutil.rmtree(config["checkpoint_dir"])

        bm.create_dirs([config["summary_dir"], config["checkpoint_dir"], config["results_dir"]])

        with open(os.path.join(config["summary_dir"], "config_dict.txt"), 'w') as f:
            f.write(json.dumps(config))

        data = None
        if dataset == "mnist":
            data = dg.DataGeneratorMNIST(config)
        elif dataset == "mocap":
            data = dg.DataGeneratorMocap(config)
        elif dataset == "cmu_walk":
            data = dg.DataGeneratorCMUWalk(config)
        elif dataset == "celebA":
            data = dg.DataGeneratorCelebA(config)

        if args.multi:
            with Pool(1) as p:
                p.apply(run, (config, model, trainer, predictor, data, args))
        else:
            run(config, model, trainer, predictor, data, args)


if __name__ == "__main__":
    main()
