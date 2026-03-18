import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper, MVSplat
    
def load_model(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, _ = get_encoder()
    decoder = get_decoder()
    model = MVSplat(encoder=encoder, decoder=decoder)
    weights = torch.load(ckpt_path, map_location='cpu')["state_dict"]
    model.load_state_dict(weights)
    model.to(device)
    return model

if __name__ == "__main__":
    print("Loading model")
    load_model("/workspace/checkpoints/re10k.ckpt")
    print("Loaded")
