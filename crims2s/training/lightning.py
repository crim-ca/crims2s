from crims2s.transform import t2m_to_normal
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .model.bayes import BayesianUpdateModel
from .model.util import DistributionToTerciles


def rps(pred, target, dim=0):
    pred_nan_mask = pred.isnan()
    target_nan_mask = target.isnan()

    rps = torch.where(
        pred_nan_mask | target_nan_mask,
        torch.zeros_like(pred),
        torch.square(
            torch.where(pred_nan_mask, torch.zeros_like(pred), pred)
            - torch.where(target_nan_mask, torch.zeros_like(target), target)
        ),
    ).sum(dim=dim)

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

        self.log("LL_Epoch/All/Train", total_ll, on_epoch=True, on_step=False)
        self.log("LL_Step/All/Train", total_ll, on_epoch=False, on_step=True)

        self.log("LL_Epoch/T2M/Train", t2m_ll, on_epoch=True, on_step=False)
        self.log("LL_Step/T2M/Train", t2m_ll, on_epoch=False, on_step=True)

        self.log("LL_Epoch/TP/Train", tp_ll, on_epoch=True, on_step=False)
        self.log("LL_Step/TP/Train", t2m_ll, on_epoch=False, on_step=True)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, batch["edges_t2m"])
        tp_terciles = self.tp_to_terciles(tp_dist, batch["edges_tp"])

        if len(batch["terciles_t2m"] == 5):
            rps_dim = 1
        else:
            rps_dim = 0

        t2m_rps = rps(t2m_terciles, batch["terciles_t2m"], dim=rps_dim)
        t2m_rps = t2m_rps.mean()

        tp_rps = rps(tp_terciles, batch["terciles_tp"], dim=rps_dim)
        tp_rps = tp_rps.mean()

        loss = t2m_rps + tp_rps

        self.log(
            "Loss_Epoch/All/Train", loss, logger=False, on_epoch=True, on_step=False
        )
        self.log(
            "Loss_Step/All/Train", loss, logger=False, on_epoch=False, on_step=True
        )

        self.log("RPS_Epoch/All/Train", loss, on_epoch=True, on_step=False)
        self.log("RPS_Step/All/Train", loss, on_epoch=False, on_step=True)

        self.log("RPS_Epoch/T2M/Train", t2m_rps, on_epoch=True, on_step=False)
        self.log("RPS_Step/T2M/Train", t2m_rps, on_epoch=False, on_step=True)

        self.log("RPS_Epoch/TP/Train", tp_rps, on_epoch=True, on_step=False)
        self.log("RPS_Step/TP/Train", tp_rps, on_epoch=False, on_step=True)

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

        total_ll = t2m_ll + tp_ll

        self.log("LL_Epoch/All/Val", total_ll, on_epoch=True, on_step=False)
        self.log("LL_Step/All/Val", total_ll, on_epoch=False, on_step=True)

        self.log("LL_Epoch/T2M/Val", t2m_ll, on_epoch=True, on_step=False)
        self.log("LL_Step/T2M/Val", t2m_ll, on_epoch=False, on_step=True)

        self.log("LL_Epoch/TP/Val", tp_ll, on_epoch=True, on_step=False)
        self.log("LL_Step/TP/Val", t2m_ll, on_epoch=False, on_step=True)

        t2m_terciles = self.t2m_to_terciles(t2m_dist, batch["edges_t2m"])
        tp_terciles = self.tp_to_terciles(tp_dist, batch["edges_tp"])

        if len(batch["terciles_t2m"] == 5):
            rps_dim = 1
        else:
            rps_dim = 0

        t2m_rps = rps(t2m_terciles, batch["terciles_t2m"], dim=rps_dim)
        t2m_rps = t2m_rps.mean()

        tp_rps = rps(tp_terciles, batch["terciles_tp"], dim=rps_dim)
        tp_rps = tp_rps.mean()

        loss = t2m_rps + tp_rps

        self.log("val_loss", loss, logger=False, on_epoch=True, on_step=False)

        self.log("Loss_Epoch/Val", loss, on_epoch=True, on_step=False)
        self.log("RPS_Epoch/AllVal", loss, on_epoch=True, on_step=False)
        self.log("RPS_Epoch/T2M/Val", t2m_rps, on_epoch=True, on_step=False)
        self.log("RPS_Epoch/TP/Val", tp_rps, on_epoch=True, on_step=False)

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
    def __init__(
        self,
        model=None,
        optimizer=None,
        scheduler=None,
        reweight_by_area=True,
        ignore_dry_tiles=False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.reweight_by_area = reweight_by_area
        self.ignore_dry_tiles = ignore_dry_tiles

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        t2m_terciles, tp_terciles = self.forward(batch)

        loss = self.compute_fields_loss(batch, t2m_terciles, tp_terciles)
        self.log("Loss_Step/All/Train", loss, on_epoch=False, on_step=True)
        self.log("Loss_Epoch/All/Train", loss, on_epoch=True, on_step=False)

        self.log_rpss(t2m_terciles, tp_terciles, batch)

        return loss

    def validation_step(self, batch, batch_id):
        t2m_terciles, tp_terciles = self.forward(batch)

        fields_loss = self.compute_fields_loss(
            batch, t2m_terciles, tp_terciles, label="Val"
        )

        self.log("val_loss", fields_loss, logger=False, on_step=False, on_epoch=True)
        self.log("Loss_Epoch/All/Val", fields_loss, on_epoch=True, on_step=False)

        self.log_rpss(t2m_terciles, tp_terciles, batch, label="Val")

        return {}

    def log_rpss(self, t2m_terciles, tp_terciles, batch, label="Train"):
        t2m_weight_mask = self.make_weight_mask(batch)
        tp_weight_mask = self.make_weight_mask(
            batch, use_dry_mask=self.ignore_dry_tiles
        )

        rpss_t2m = self.compute_rpss(
            batch, t2m_terciles, batch["terciles_t2m"], t2m_weight_mask
        )
        rpss_tp = self.compute_rpss(
            batch, tp_terciles, batch["terciles_tp"], tp_weight_mask
        )

        rpss = rpss_t2m + rpss_tp
        self.log(f"RPSS_Epoch/All/{label}", rpss, on_epoch=True, on_step=False)
        self.log(f"RPSS_Epoch/T2M/{label}", rpss_t2m, on_epoch=True, on_step=False)
        self.log(f"RPSS_Epoch/TP/{label}", rpss_tp, on_epoch=True, on_step=False)

    def make_weight_mask(self, batch, use_dry_mask=False):
        weight_mask = torch.ones(121, 240, device=batch["obs_t2m"].device)

        latitude = batch["latitude"]

        if self.reweight_by_area:
            if len(latitude.shape) == 2:
                latitude = latitude[0]
            lat_weights = self.make_lat_weights(latitude)
            weight_mask *= lat_weights

        if use_dry_mask:
            dry_mask = batch["dry_mask_tp"]

            if len(latitude.shape) == 2:
                dry_mask = dry_mask[0]
            dry_weights = self.make_dry_weights(dry_mask)
            weight_mask *= dry_weights

        return weight_mask

    def make_lat_weights(self, latitude):
        lat_weights = torch.ones(121, 240, device=latitude.device)
        lat_mask = latitude <= -60.0
        lat_weights[lat_mask, :] = 0.0

        area_by_lat = torch.cos(torch.deg2rad(torch.abs(latitude))).unsqueeze(-1)

        lat_weights *= area_by_lat

        return lat_weights

    def make_dry_weights(self, dry_mask):
        dry_weights = torch.ones(121, 240, device=dry_mask.device, dtype=float)
        dry_weights[dry_mask] = 0.0

        return dry_weights

    def compute_fields_loss(self, batch, t2m_terciles, tp_terciles, label="Train"):
        t2m_weight_mask = self.make_weight_mask(batch)
        tp_weight_mask = self.make_weight_mask(
            batch, use_dry_mask=self.ignore_dry_tiles
        )

        t2m_rps = self.compute_rps(t2m_terciles, batch["terciles_t2m"])
        tp_rps = self.compute_rps(tp_terciles, batch["terciles_tp"])

        t2m_rps = (t2m_rps * t2m_weight_mask).mean()
        tp_rps = (tp_rps * tp_weight_mask).mean()

        loss = t2m_rps + tp_rps

        self.log(f"RPS_Epoch/All/{label}", loss, on_epoch=True, on_step=False)
        self.log(f"RPS_Epoch/T2M/{label}", t2m_rps, on_epoch=True, on_step=False)
        self.log(f"RPS_Epoch/TP/{label}", tp_rps, on_epoch=True, on_step=False)

        return loss

    def compute_rps(self, model, target):
        if len(target.shape) == 5:
            rps_dim = 1
        else:
            rps_dim = 0

        rps_score = rps(model, target, dim=rps_dim)
        return rps_score

    def compute_rpss(self, batch, model, target, weight_mask):
        climatology = torch.full_like(target, 0.33)

        model_rps = self.compute_rps(model, target)
        climatology_rps = self.compute_rps(climatology, target)

        aggregated_model_rps = model_rps.mean(dim=-3)
        aggregated_climatology_rps = climatology_rps.mean(dim=-3)

        rpss = 1.0 - aggregated_model_rps / aggregated_climatology_rps

        rpss *= weight_mask
        rpss_nan_mask = rpss.isnan()

        weight_zero = weight_mask == 0.0

        return rpss[~rpss_nan_mask & ~weight_zero].mean()

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


class S2SBayesModelModule(S2STercilesModule):
    def __init__(
        self,
        model: BayesianUpdateModel,
        optimizer,
        scheduler,
        regularization: float,
        model_only_epochs=0,
        **kwargs,
    ):
        super().__init__(model, optimizer, scheduler, **kwargs)
        self.regularization = regularization
        self.model_only_epochs = model_only_epochs

    def on_epoch_start(self) -> None:
        if self.current_epoch < self.model_only_epochs:
            for p in self.model.t2m_weight_model.parameters():
                p.requires_grad = False

            for p in self.model.tp_weight_model.parameters():
                p.requires_grad = False

        else:
            for p in self.model.t2m_weight_model.parameters():
                p.requires_grad = True

            for p in self.model.tp_weight_model.parameters():
                p.requires_grad = True

    def training_step(self, batch, batch_idx):
        t2m_terciles, tp_terciles, t2m_prior_weights, tp_prior_weights = self.forward(
            batch
        )
        fields_loss = self.compute_fields_loss(batch, t2m_terciles, tp_terciles)
        reg_loss = self.compute_reg_loss(t2m_prior_weights, tp_prior_weights)

        loss = fields_loss + reg_loss

        self.log("Loss_Epoch/All/Train", loss, on_epoch=True, on_step=False)
        self.log("Loss_Step/All/Train", loss, on_epoch=False, on_step=True)

        self.log(
            "Loss_Epoch/PriorWeights/Train", reg_loss, on_epoch=True, on_step=False
        )
        self.log("Loss_Step/PriorWeights/Train", reg_loss, on_epoch=False, on_step=True)
        self.log("Loss_Epoch/Fields/Train", fields_loss, on_epoch=True, on_step=False)

        self.log(
            "PriorWeights_Epoch/T2M/Train",
            t2m_prior_weights.detach().mean(),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PriorWeights_Epoch/TP/Train",
            tp_prior_weights.detach().mean(),
            on_epoch=True,
            on_step=False,
        )

        prior_weights = torch.cat([t2m_prior_weights, tp_prior_weights])
        prior_weights_mean = prior_weights.detach().mean()
        self.log(
            "PriorWeights_Epoch/All/Train",
            prior_weights_mean,
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PriorWeights_Step/All/Train",
            prior_weights_mean,
            on_epoch=False,
            on_step=True,
        )

        self.log_rpss(t2m_terciles, tp_terciles, batch, label="Train")

        return loss

    def compute_reg_loss(self, t2m_prior_weights, tp_prior_weights):
        prior_weights = torch.cat([t2m_prior_weights, tp_prior_weights])
        square_weights = torch.square(prior_weights)

        reg_loss = self.regularization * square_weights.mean()

        return reg_loss

    def validation_step(self, batch, batch_id):
        t2m_terciles, tp_terciles, t2m_prior_weights, tp_prior_weights = self.forward(
            batch
        )

        fields_loss = self.compute_fields_loss(
            batch, t2m_terciles, tp_terciles, label="Val"
        )

        reg_loss = self.compute_reg_loss(t2m_prior_weights, tp_prior_weights)

        loss = fields_loss + reg_loss

        self.log("val_loss", loss, logger=False, on_epoch=True, on_step=False)

        self.log("Loss_Epoch/All/Val", loss, on_epoch=True, on_step=False)

        self.log("Loss_Epoch/Fields/Val", fields_loss, on_epoch=True, on_step=False)
        self.log("Loss_Epoch/PriorWeights/Val", reg_loss, on_epoch=True, on_step=False)
        self.log(
            "PriorWeights_Epoch/T2M/Val",
            t2m_prior_weights.detach().mean(),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PriorWeights_Epoch/TP/Val",
            tp_prior_weights.detach().mean(),
            on_epoch=True,
            on_step=False,
        )
        self.log(
            "PriorWeights_Epoch/All/Val",
            torch.cat([t2m_prior_weights, tp_prior_weights]).detach().mean(),
            on_epoch=True,
            on_step=False,
        )

        self.log_rpss(t2m_terciles, tp_terciles, batch, label="Val")

        return {}
