import os
from pathlib import Path
import warnings

import torch
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

import json

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
    from src.dataset.types import BatchedExample, BatchedViews
    
def load_images(images_path):
    images_path = Path(images_path)
    with open(images_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    images = []
    for img_path in metadata['image_paths']:
        full_path = images_path / img_path
        img = Image.open(full_path).convert('RGB')
        img_tensor = tf.to_tensor(img)
        img_tensor = tf.Resize((256, 256))(img_tensor)
        images.append(img_tensor)
    
    images = torch.stack(images)
    target_images = torch.zeros((len(metadata['extrinsics']) - 2, images.shape[1], images.shape[2], images.shape[3]))
    
    extrinsics_list = []
    for ext in metadata['extrinsics']:
        extrinsics_list.append(torch.tensor(ext, dtype=torch.float32))
        
    extrinsics = torch.stack(extrinsics[:2])
    target_extrinsics = torch.stack(extrinsics[2:])
    
    intrinsics_list = []
    for _ in range(len(metadata['image_paths'])):
        intrinsics_list.append(torch.tensor(metadata['intrinsics'], dtype=torch.float32))
    intrinsics = torch.stack(intrinsics_list[:2])
    target_intrinsics = torch.stack(intrinsics_list[2:])
    
    context_views_tensor = torch.tensor([0, 1])
    target_context_views_tensor = torch.tensor([range(2, len(extrinsics_list))])
    
    src_example: BatchedViews = {
        "extrinsics": extrinsics.unsqueeze(0) ,
        "intrinsics": intrinsics.unsqueeze(0) ,
        "image": images.unsqueeze(0) ,
        "near": torch.ones((1, 2)),
        "far": 100.0 * torch.ones((1, 2)),
        'index': context_views_tensor.unsqueeze(0) 
    }
    
    tgt_example: BatchedViews = {
        "extrinsics": target_extrinsics.unsqueeze(0) ,
        "intrinsics": target_intrinsics.unsqueeze(0) ,
        "image": target_images.unsqueeze(0) ,
        "near": torch.ones((1, 2)),
        "far": 100.0 * torch.ones((1, 2)),
        'index': target_context_views_tensor.unsqueeze(0) 
    }
    
    example: BatchedExample = {
        "context" : src_example,
        "target" : tgt_example,
        "scene" : ["scene_001"]
    }
    
    return example
    
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
    
    print("Loading images")
    images_dir = "/workspace/my_photos/"
    
    data = load_images(images_dir)
    print("Loaded images")
    
