from crims2s.transform import t2m_to_normal
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .model.util import DistributionToTerciles


def rps(pred, target, dim=0):
    pred_nan_mask = pred.isnan()
    target_nan_mask = target.isnan()

    rps = torch.where(
        pred_nan_mask | target_nan_mask,
        torch.zeros_like(pred),
        torch.square(
            torch.where(pred_nan_mask, torch.zeros_like(pred), pred)
            - torch.where(target_nan_mask, torch.zeros_like(target), target).sum(
                dim=dim
            )
        ),
    )

    return rps


class S2SDistributionModule(pl.LightningModule):
    # We specify default values for model and optimizer even though it doesn't make sense.
    # We do this so that the module can be brought back to life with load_from_checkpoint.
    def __init__(self, model=None, optimizer=None, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.t2m_to_terciles = DistributionToTerciles()
        self.tp_to_terciles = DistributionToTerciles()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        t2m_dist, tp_dist = self.forward(batch)

        t2m_ll = self.compute_negative_log_likelihood(t2m_dist, batch["obs_t2m"])
        tp_ll = self.compute_negative_log_likelihood(
            tp_dist, batch["obs_tp"], regularization=1e-9
        )

        total_ll = t2m_ll + tp_ll

        self.log("LL/All/Train", total_ll, on_epoch=True, on_step=True)
        self.log("LL/T2M/Train", t2m_ll, on_epoch=True, on_step=True)
        self.log("LL/TP/Train", tp_ll, on_epoch=True, on_step=True)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, batch["edges_t2m"])
        tp_terciles = self.tp_to_terciles(tp_dist, batch["edges_tp"])

        t2m_rps = rps(t2m_terciles, batch["terciles_t2m"])
        t2m_rps = t2m_rps.mean()

        tp_rps = rps(tp_terciles, batch["terciles_tp"])
        tp_rps = tp_rps.mean()

        loss = t2m_rps + tp_rps

        self.log("RPS/All/Train", loss, on_epoch=True, on_step=True)
        self.log("RPS/T2M/Train", t2m_rps, on_epoch=True, on_step=True)
        self.log("RPS/TP/Train", tp_rps, on_epoch=True, on_step=True)

        return loss

    def compute_negative_log_likelihood(self, dist, obs, regularization=0.0):
        nan_mask = obs.isnan()
        obs[nan_mask] = 0.0

        log_likelihood = dist.log_prob(obs + regularization)
        log_likelihood[nan_mask] = 0.0

        return -log_likelihood.mean()

    def validation_step(self, batch, batch_id):
        t2m_dist, tp_dist = self.forward(batch)

        t2m_ll = self.compute_negative_log_likelihood(t2m_dist, batch["obs_t2m"])
        tp_ll = self.compute_negative_log_likelihood(
            tp_dist, batch["obs_tp"], regularization=1e-9
        )

        ll_all = t2m_ll + tp_ll

        self.log("LL/All/Val", ll_all, on_epoch=True, on_step=True)
        self.log("LL/T2M/Val", t2m_ll, on_epoch=True, on_step=True)
        self.log("LL/TP/Val", tp_ll, on_epoch=True, on_step=True)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, batch["edges_t2m"])
        tp_terciles = self.tp_to_terciles(tp_dist, batch["edges_tp"])

        t2m_rps = rps(t2m_terciles, batch["terciles_t2m"])
        t2m_rps = t2m_rps.mean()

        tp_rps = rps(tp_terciles, batch["terciles_tp"])
        tp_rps = tp_rps.mean()

        loss = t2m_rps + tp_rps

        self.log("val_loss", loss, logger=False, on_epoch=True, on_step=False)
        self.log("RPS/All/Val", loss, on_epoch=True, on_step=True)
        self.log("RPS/T2M/Val", t2m_rps, on_epoch=True, on_step=True)
        self.log("RPS/TP/Val", tp_rps, on_epoch=True, on_step=True)

        return {}

    def configure_optimizers(self):
        return_dict = {
            "optimizer": self.optimizer,
        }

        if self.scheduler is not None:
            return_dict["lr_scheduler"] = {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
            }

        return return_dict


class S2STercilesModule(pl.LightningModule):
    """Lightning module for models that output terciles directly, instead of modules
    that output distributions."""

    # We specify default values for model and optimizer even though it doesn't make sense.
    # We do this so that the module can be brought back to life with load_from_checkpoint.
    def __init__(self, model=None, optimizer=None, scheduler=None):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        t2m_terciles, tp_terciles = self.forward(batch)

        t2m_rps = rps(t2m_terciles, batch["terciles_t2m"])
        t2m_rps = t2m_rps.mean()

        tp_rps = rps(tp_terciles, batch["terciles_tp"])
        tp_rps = tp_rps.mean()

        loss = t2m_rps + tp_rps

        self.log("RPS/All/Train", loss, on_epoch=True, on_step=True)
        self.log("RPS/T2M/Train", t2m_rps, on_epoch=True, on_step=True)
        self.log("RPS/TP/Train", tp_rps, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_id):
        t2m_terciles, tp_terciles = self.forward(batch)

        t2m_rps = rps(t2m_terciles, batch["terciles_t2m"])
        t2m_rps = t2m_rps.mean()

        tp_rps = rps(tp_terciles, batch["terciles_tp"])
        tp_rps = tp_rps.mean()

        loss = t2m_rps + tp_rps

        self.log("val_loss", loss, logger=False, on_epoch=True, on_step=False)
        self.log("RPS/All/Val", loss, on_epoch=True, on_step=True)
        self.log("RPS/T2M/Val", t2m_rps, on_epoch=True, on_step=True)
        self.log("RPS/TP/Val", tp_rps, on_epoch=True, on_step=True)

        return {}

    def configure_optimizers(self):
        return_dict = {
            "optimizer": self.optimizer,
        }

        if self.scheduler is not None:
            return_dict["lr_scheduler"] = {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
            }

        return return_dict
