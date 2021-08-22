# CRIMS2S Training

The model training logic for the S2S AI challenge.

```
s2s_train model.biweekly=False
```

Use hydra laucher to run multiple experiments simultaneously
```
s2s_train hydra/launcher=submitit_slurm optimizer=adam,sgd model.biweekly=True,False devices=1 +hydra.launcher.additional_parameters.gres="gpu:1" -m
```