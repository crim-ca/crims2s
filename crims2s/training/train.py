import collections.abc
import hydra
import logging
import os
import pytorch_lightning as pl
import pytorch_lightning.callbacks as pl_callbacks
import pytorch_lightning.loggers as loggers
import torch

from ..dataset import S2SDataset, TransformedDataset
from ..util import ECMWF_FORECASTS

_logger = logging.getLogger(__name__)


class ModelCheckpoint(pl_callbacks.ModelCheckpoint):
    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint
    ):
        if self.best_model_score is not None:
            pl_module.log(f"{self.monitor}_min", self.best_model_score)
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)


def make_datasets(dataset_cfg, transform_cfg):
    transform = hydra.utils.instantiate(transform_cfg)

    train_years = list(range(2000, dataset_cfg.validate_from))
    val_years = list(range(dataset_cfg.validate_from, 2020))

    if dataset_cfg.index is not None:
        month, day = ECMWF_FORECASTS[dataset_cfg.index]
        label = f"{month:02}{day:02}.nc"

        _logger.info("Targetting monthday %s", label)
        name_filter = lambda x: x.endswith(label)
    else:
        name_filter = None

    train_dataset = TransformedDataset(
        S2SDataset(
            dataset_cfg.dataset_dir,
            years=train_years,
            name_filter=name_filter,
            include_features=dataset_cfg.include_features,
        ),
        transform,
    )
    val_dataset = TransformedDataset(
        S2SDataset(
            dataset_cfg.dataset_dir,
            years=val_years,
            name_filter=name_filter,
            include_features=dataset_cfg.include_features,
        ),
        transform,
    )

    return train_dataset, val_dataset


def run_experiment(cfg, num_workers=4, lr_find=False):
    train_dataset, val_dataset = make_datasets(cfg.dataset, cfg.transform)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        batch_sampler=None,
        num_workers=num_workers,
        shuffle=True,
        drop_last=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        batch_sampler=None,
        num_workers=num_workers,
        drop_last=False,
    )

    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.call(cfg.optimizer, model)

    if "scheduler" in cfg:
        scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer)
    else:
        scheduler = None

    lightning_module = hydra.utils.instantiate(cfg.module, model, optimizer, scheduler)
    tensorboard = loggers.TensorBoardLogger("./tensorboard", default_hp_metric=False)

    mlflow = loggers.MLFlowLogger(
        cfg.logging.experiment_name,
        run_name=cfg.logging.run_name,
        tracking_uri=cfg.logging.mlflow_uri,
        tags={
            "user": cfg.user,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
            "cwd": os.getcwd(),
            "original_cwd": hydra.utils.get_original_cwd(),
        },
    )
    mlflow.log_hyperparams(cfg)

    checkpointer = ModelCheckpoint(monitor="val_loss")

    callbacks = [
        checkpointer,
        pl_callbacks.LearningRateMonitor(),
    ]
    if "early_stopping" in cfg:
        early_stopping = pl_callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.early_stopping.patience
        )
        callbacks.append(early_stopping)

    trainer = pl.Trainer(
        log_every_n_steps=1,
        logger=[tensorboard, mlflow],
        callbacks=callbacks,
        default_root_dir="./lightning/",
        **cfg.trainer,
    )

    if lr_find:
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
        mlflow.log_metrics({"val_loss_min": best_score})


@hydra.main(config_path="conf", config_name="config")
def cli(cfg):
    run_experiment(
        cfg.experiment, num_workers=int(cfg.num_workers), lr_find=cfg.lr_find
    )


if __name__ == "__main__":
    cli()
