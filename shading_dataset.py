#@title Dataset

from pathlib import Path
from torchvision import transforms
from PIL import Image

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage
from skimage import color as skimage_color # For robust RGB to Lab
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
import kornia
def collate_fn(examples):
    gscale_values = [example["input_pixels"] for example in examples]
    # rgb_values = [example["rgb_pixels"] for example in examples]
    controlnet_input_values = [example["controlnet_input_pixels"] for example in examples]

    controlnet_input_values = torch.stack(controlnet_input_values)
    controlnet_input_values = controlnet_input_values.to(memory_format=torch.contiguous_format).float()

    gscale_values = torch.stack(gscale_values)
    gscale_values = gscale_values.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_pixels": gscale_values,
        "controlnet_input_pixels": controlnet_input_values 
    }
    return batch


class ShadingDataset:
    def __init__(self, instance_data_root, device, H=512, W=512, size=100, batch_size=1):
        super().__init__()

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        self.instance_images_path = list(Path(instance_data_root).iterdir())
        self.instance_images_path = self.instance_images_path[:min(len(self.instance_images_path), size)]
        self.num_instance_images = 32
        self._length = self.num_instance_images
        self.device = device
        self.H = H
        self.W = W
        self.size = size
        self.batch_size = batch_size
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.controlnet_image_transforms = transforms.Compose(
            [
                transforms.Resize(size=(H, W)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["input_pixels"] = self.image_transforms(instance_image).type(torch.float16).to(self.device)
        example["controlnet_input_pixels"] = self.controlnet_image_transforms(instance_image).type(torch.float16).to(self.device)

        return example


    def dataloader(self):
        loader = DataLoader(self, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=false, num_workers=0)
        return loader



class SingleImageDataset:
    def __init__(self, instance_data_path, device, size=1, H=512, W=512):
        super().__init__()

        self.instance_image_path = Path(instance_data_path)
        self.num_instance_images = 1
        self._length = self.num_instance_images
        self.device = device
        self.H = H
        self.W = W
        self.size = size
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((H, W), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )
        self.controlnet_image_transforms = transforms.Compose(
            [
                transforms.Resize(size=(H, W)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def decolor(self, img):
      return rgb_to_grayscale(img, num_output_channels=3)


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_image_path)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        example["input_pixels"] = self.image_transforms(instance_image).type(torch.float16).to(self.device)
        example["controlnet_input_pixels"] = self.controlnet_image_transforms(instance_image).type(torch.float16).to(self.device)
    
        return example


    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=1, collate_fn=collate_fn, shuffle=false, num_workers=0)
        return loader
    
    
class ColorizationDataset(Dataset):
    def __init__(self, instance_data_path, device, H=512, W=512):
        self.device = device
        self.instance_image_path = instance_data_path
        # Target size for U-Net and processing
        self.H = H 
        self.W = W

        try:
            pil_image = Image.open(self.instance_image_path)
            if not pil_image.mode == "RGB":
                pil_image = pil_image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not load image from {self.instance_image_path}: {e}")
        
        # Resize image first
        pil_image = pil_image.resize((self.W, self.H), Image.Resampling.LANCZOS)
        image_np_rgb = np.array(pil_image) # Shape (H, W, 3), values [0, 255]
        
        # Convert RGB [0, 255] to Lab. skimage.color.rgb2lab expects float input in [0, 1].
        img_tensor = torch.from_numpy(image_np_rgb).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        img_lab = kornia.color.rgb_to_lab(img_tensor) # L: [0, 100], a,b: approx [-128, 127]
        l_channel = img_lab[0, 0:1, :, :]
        # L channel for U-Net input: scale L from [0, 100] to [0, 1] (common for normalized inputs)
        self.l_channel_unet_input_tensor = l_channel / 100.0 
        # Convert to tensor: (1, H, W)
        # L channel for composition (will be scaled back to [0,100] before Lab->RGB conversion)
        # This is the same as l_channel_unet_input, also (1, H, W)
        # self.l_channel_for_composition_tensor = self.l_channel_unet_input_tensor.clone()

        # # Control image for ControlNet: L channel replicated to 3 channels, normalized to [0,1]
        # # ControlNet expects a 3-channel image.
        # control_img_np = np.stack([self.l_channel_unet_input]*3, axis=-1) # (H, W, 3) with L values [0,1]
        # self.control_image_tensor = transforms.ToTensor()(control_img_np.astype(np.float32)) # (3, H, W)

    def __len__(self):
        return 1 # For single image colorization

    def __getitem__(self, index):
        return {
            "l_channel_unet_input": self.l_channel_unet_input_tensor.to(self.device),    # (1, H, W)            # "control_image_for_controlnet": self.control_image_tensor.to(self.device), # (3, H, W)
        }

def colorization_collate_fn(batch_list):
    # If batch_size is 1 (typical for this kind of single-image optimization)
    # If batch_size > 1 (general case)
    keys = batch_list[0].keys()
    collated_batch = {k: torch.stack([d[k] for d in batch_list]) for k in keys}
    return collated_batch