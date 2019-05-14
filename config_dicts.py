def config_dict(parent_folder, vae_q, dataset):
    config = {}

    if dataset == "mnist":
        config = {
            "output_distribution": "bernoulli",
            "batch_size": 1000,
            "num_data_points": 60000,
            "state_size": [28, 28],
            "label_latent_manifold": True,
            "plot_dimensions": 2,
            "learning_rate": 0.01
        }
    elif dataset == "mocap":
        config = {
            "output_distribution": "gaussian",
            "batch_size": 31,
            "num_data_points": 217,
            "state_size": [123],
            "label_latent_manifold": False,
            "plot_dimensions": 3,
            "learning_rate": 0.01
        }
    elif dataset == "cmu_walk":
        config = {
            "output_distribution": "gaussian",
            "batch_size": 257,
            "num_data_points": 4369,
            "state_size": [123],
            "label_latent_manifold": False,
            "plot_dimensions": 3,
            "learning_rate": 0.01
        }
    elif dataset == "mixture_gauss":
        config = {
            "output_distribution": "gaussian",
            "batch_size": 50,
            "num_data_points": 1000,
            "state_size": [2],
            "label_latent_manifold": False,
            "plot_dimensions": 1,
            "learning_rate": 0.01
        }
    else:
        print("Invalid dataset")

    # Folders where everything is saved
    config["summary_dir"] = f"{parent_folder}/Z{vae_q}_summary/"
    config["results_dir"] = f"{parent_folder}/Z{vae_q}_results/"
    config["checkpoint_dir"] = f"{parent_folder}/Z{vae_q}_checkpoint/"
    # Max number of checkpoints to keep
    config["max_to_keep"] = 5
    # VAE latent dimensions
    config["vae_q"] = vae_q
    # Number of draws used in marginal KL calculation and to get the test metrics
    config["num_draws"] = 480
    # Max training epochs
    config["num_epochs"] = 500
    # Number of points in the X latent space from which to get reconstructed images
    config["num_plot_x_points"] = 30
    config["num_geodesics"] = 5
    # Max value in each dimension X latent space from which to generate images
    config["max_x_value"] = 1.5
    # Number of iterations per epoch
    config["num_iter_per_epoch"] = config["num_data_points"] // config["batch_size"]

    return config
