import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class S2SLightningModule(pl.LightningModule):
    # We specify default values for model and optimizer even though it doesn't make sense.
    # We do this so that the module can be brought back to life with load_from_checkpoint.
    def __init__(self, model=None, optimizer=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        t2m_dist, tp_dist = self.forward(batch)

        t2m_loss = self.compute_negative_log_likelihood(t2m_dist, batch["obs_t2m"])
        tp_loss = self.compute_negative_log_likelihood(
            tp_dist, batch["obs_tp"], regularization=1e-9
        )

        loss = t2m_loss + tp_loss

        self.log("LL/All/Train", loss, on_epoch=True, on_step=True)
        self.log("LL/T2M/Train", t2m_loss, on_epoch=True, on_step=True)
        self.log("LL/TP/Train", tp_loss, on_epoch=True, on_step=True)

        return loss

    def compute_negative_log_likelihood(self, dist, obs, regularization=0.0):
        nan_mask = obs.isnan()
        obs[nan_mask] = 0.0

        log_likelihood = dist.log_prob(obs + regularization)
        log_likelihood[nan_mask] = 0.0

        return -log_likelihood.mean()

    def validation_step(self, batch, batch_id):
        t2m_dist, tp_dist = self.forward(batch)

        t2m_loss = self.compute_negative_log_likelihood(t2m_dist, batch["obs_t2m"])
        tp_loss = self.compute_negative_log_likelihood(
            tp_dist, batch["obs_tp"], regularization=1e-9
        )

        loss = t2m_loss + tp_loss

        self.log("val_loss", loss, logger=False, on_epoch=True, on_step=False)
        self.log("LL/All/Val", loss, on_epoch=True, on_step=True)
        self.log("LL/T2M/Val", t2m_loss, on_epoch=True, on_step=True)
        self.log("LL/TP/Val", tp_loss, on_epoch=True, on_step=True)

        return {}

        # return {
        #     "LL/All/Val": loss.detach(),
        #     "LL/T2M/Val": t2m_loss.detach(),
        #     "LL/TP/Val": tp_loss.detach(),
        # }

    def configure_optimizers(self):
        return self.optimizer
