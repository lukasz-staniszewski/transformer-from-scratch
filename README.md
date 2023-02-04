# Implementation of Transformer from scratch using PyTorch Lightning

## I. Features
+ Code organisation using [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).
+ Data loading using `datamodules` from [PyTorch Lightning Bolts](https://lightning-bolts.readthedocs.io/en/latest/>).
+ Experiments configuration using [Hydra](https://hydra.cc/docs/next/tutorials/intro/>).
+ Training visualisation using [Weights & Biases (wandb.ai)](https://docs.wandb.ai/).

## II. Installation
Create pip environment:

    $ pip install -r requirements.txt

Configure environment in ``.env`` for wandb.ai::

    DATA_DIR=datasets
    RESULTS_DIR=results
    WANDB_ENTITY=WANDB_LOGIN
    WANDB_PROJECT=WANDB_PROJECT_NAME

``DATA_DIR`` - directory for data storage. ``RESULTS_DIR`` - directory for results.


## III. Running experiments
Running training from main dir:

    $ python -m transformer.main

Adding metadata:
    
    $ python -m transformer.main notes="First training" tags="[TAG1, TAG2]"

Change of single settings:

    $ python -m transformer.main pl.max_epochs=150 experiment.batch_size=64

Change of whole settings based on YAML file
(i.e. ``src/transformer/configs/experiment/fashion.yaml``):

    $ python -m transformer.main experiment=fashion

Turning off stream to *wandb*::

    $ WANDB_MODE=dryrun python -m transformer.main 

*Debug* mode (turning off *wandb* logging)::

    $ RUN_MODE=debug python -m transformer.main


## IV. Upload of checkpoint to *wandb*
[WANDB.AI](https://wandb.ai) enables the remote storage and sharing of files (i.e. trained models) using [W&B Artifacts](https://docs.wandb.ai/guides/artifacts/api).

Uploading of checkpoints generated during training process may be performed using `cli.py`::

    $ python -m transformer.cli upload RUN_NAME CHECKPOINT_NAME ARTIFACT_NAME

For example:

    $ python -m transformer.cli upload 20230202-1720-tf epoch_5.ckpt test_model

Then you are free to resume training from given checkpoint on another instance:

    resume_checkpoint: wandb://WANDB_USER/WANDB_PROJECT/test_model:v0@epoch_5.ckpt
