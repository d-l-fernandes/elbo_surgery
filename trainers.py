import numpy as np
from tqdm import tqdm

import base_model as bm

class VAETrainer(bm.BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(VAETrainer, self).__init__(sess, model, data, config, logger)
        input_shape = [self.config["batch_size"]] + self.config["state_size"]
        self.sess.run(self.init,
                      feed_dict={self.model.t_y: np.ones(input_shape)})
        self.batch_gen = None

    def train_epoch(self, cur_epoch):
        # name of metric: [variable, if mean is to be used]
        metrics = {
            "elbo": [[], True],
            "reco": [[], True],
            "hist_reco": [[], False],
            "kl": [[], True],
        }

        self.batch_gen = self.data.select_batch_generator("training")
        loop = tqdm(range(self.config["num_iter_per_epoch"]), desc=f"Epoch {cur_epoch+1}/{self.config['num_epochs']}",
                    ascii=True)

        for _ in loop:
            metrics_step = self.train_step()
            metrics = self.update_metrics_dict(metrics, metrics_step)

        summaries_dict = self.create_summaries_dict(metrics)

        self.logger.summarize(cur_epoch+1, summaries_dict=summaries_dict)
        self.model.save(self.sess, summaries_dict["Metrics/elbo"])

        return summaries_dict["Metrics/elbo"]

    def train_step(self):
        batch_y, batch_x = next(self.batch_gen)
        feed_dict = {self.model.t_y: batch_y}

        _, cost, reco, reco_full, kl = \
            self.sess.run((
                self.model.opt_trainer,
                self.model.t_avg_elbo_loss,
                self.model.t_avg_reco,
                self.model.t_full_reco,
                self.model.t_avg_kl),
                feed_dict)

        metrics = {
            "elbo": cost,
            "reco": reco,
            "hist_reco": reco_full,
            "kl": kl,
        }

        return metrics
