import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class S2SLightningModule(pl.LightningModule):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        t2m_dist, tp_dist = self.forward(batch)

        t2m_loss = self.compute_negative_log_likelihood(t2m_dist, batch["obs_t2m"])
        tp_loss = self.compute_negative_log_likelihood(tp_dist, batch["obs_tp"])

        loss = t2m_loss + tp_loss

        self.log("LL/All/Train", loss)
        self.log("LL/T2M/Train", t2m_loss)
        self.log("LL/TP/Train", tp_loss)

        return {
            "loss": loss,
            "LL/All/Train": loss.detach(),
            "LL/T2M/Train": t2m_loss.detach(),
            "LL/TP/Train": tp_loss.detach(),
        }

    def training_epoch_end(self, outs):
        for key in ["LL/All/Train", "LL/T2M/Train", "LL/TP/Train"]:
            mean_value = np.array([float(x[key]) for x in outs]).mean()
            self.log(key + "/EpochMean", mean_value)

    def compute_negative_log_likelihood(self, dist, obs):
        nan_mask = obs.isnan()
        obs[nan_mask] = 0.0
        log_likelihood = dist.log_prob(obs)
        log_likelihood[nan_mask] = 0.0

        return -log_likelihood.mean()

    def validation_step(self, batch, batch_id):
        t2m_dist, tp_dist = self.forward(batch)

        t2m_loss = self.compute_negative_log_likelihood(t2m_dist, batch["obs_t2m"])
        tp_loss = self.compute_negative_log_likelihood(tp_dist, batch["obs_tp"])

        loss = t2m_loss + tp_loss

        self.log("LL/All/Val", loss)
        self.log("LL/T2M/Val", t2m_loss)
        self.log("LL/TP/Val", tp_loss)

        return {
            "LL/All/Val": loss.detach(),
            "LL/T2M/Val": t2m_loss.detach(),
            "LL/TP/Val": tp_loss.detach(),
        }

    def validation_epoch_end(self, outs):
        for key in ["LL/All/Val", "LL/T2M/Val", "LL/TP/Val"]:
            mean_value = np.array([float(x[key]) for x in outs]).mean()
            self.log(key + "/EpochMean", mean_value)

    def configure_optimizers(self):
        return self.optimizer
