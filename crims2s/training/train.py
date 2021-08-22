import hydra
import logging
import os
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
import pytorch_lightning.loggers as loggers
import torch

from ..dataset import S2SDataset, TransformedDataset
from ..util import ECMWF_FORECASTS
from .lightning import S2SLightningModule

_logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def cli(cfg):
    transform = hydra.utils.instantiate(cfg.transform)

    train_years = list(range(2000, cfg.validate_from))
    val_years = list(range(cfg.validate_from, 2020))

    if cfg.index is not None:
        month, day = ECMWF_FORECASTS[cfg.index]
        label = f"{month:02}{day:02}.nc"

        _logger.info("Targetting monthday %s", label)
        name_filter = lambda x: x.endswith(label)
    else:
        name_filter = None

    train_dataset = TransformedDataset(
        S2SDataset(
            cfg.dataset_dir,
            years=train_years,
            name_filter=name_filter,
            include_features=False,
        ),
        transform,
    )
    val_dataset = TransformedDataset(
        S2SDataset(
            cfg.dataset_dir,
            years=val_years,
            name_filter=name_filter,
            include_features=False,
        ),
        transform,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=None,
        batch_sampler=None,
        num_workers=int(cfg.num_workers),
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=None,
        batch_sampler=None,
        num_workers=int(cfg.num_workers),
    )

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())

    lightning_module = S2SLightningModule(model, optimizer)

    tensorboard = loggers.TensorBoardLogger("./tensorboard", default_hp_metric=False)

    mlflow = loggers.MLFlowLogger(
        cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_uri,
        tags={"user": cfg.user},
    )
    mlflow.log_hyperparams(cfg)

    early_stopping = callbacks.EarlyStopping(monitor="val_loss")
    checkpointer = callbacks.ModelCheckpoint(monitor="val_loss")

    trainer = pl.Trainer(
        devices=cfg.devices,
        accelerator=cfg.accelerator,
        max_epochs=cfg.max_epochs,
        log_every_n_steps=1,
        logger=[tensorboard, mlflow],
        callbacks=[early_stopping, checkpointer],
        default_root_dir="./lightning/",
    )

    if cfg.lr_find:
        import matplotlib.pyplot as plt

        lr_finder = trainer.tuner.lr_find(
            lightning_module, train_dataloader, val_dataloader
        )
        lr_finder.results
        fig = lr_finder.plot(suggest=True)
        filename = "lr.png"
        plt.savefig(filename)
        _logger.info(f"Saved LR curve: {os.getcwd() + '/' + filename}.")
    else:
        trainer.fit(lightning_module, train_dataloader, val_dataloader)
        best_score = float(checkpointer.best_model_score.cpu())
        mlflow.log_metrics({"min_val_loss": best_score})


if __name__ == "__main__":
    cli()
