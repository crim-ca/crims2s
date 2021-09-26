import logging
import torch
import pytorch_lightning
import pytorch_lightning.callbacks

from .lightning import S2SBayesModelModule


_logger = logging.getLogger(__name__)


class BayesForecastFreezeUnfreezeCallback(pytorch_lightning.callbacks.BaseFinetuning):
    """Callback that controls when we optimize the forecast models. It is useful because
    it allows us to train the forecast models for a couple epochs before trying to 
    weight them."""

    def __init__(self, unfreeze_weights_epoch=0):
        super().__init__()
        self.unfreeze_weights_epoch = unfreeze_weights_epoch

    def freeze_before_training(self, pl_module: S2SBayesModelModule):
        self.freeze(pl_module.model.weight_model, train_bn=False)

    def finetune_function(
        self,
        pl_module: S2SBayesModelModule,
        current_epoch,
        optimizer: torch.optim.Optimizer,
        optimizer_idx,
    ):
        if current_epoch == self.unfreeze_weights_epoch:
            _logger.info("Thawing weight model.")

            self.unfreeze_and_add_param_group(
                modules=pl_module.model.weight_model,
                optimizer=optimizer,
                train_bn=True,
            )
