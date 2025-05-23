import random
import os
import argparse
import numpy as np
import torch
from PIL import Image
import glob
import torch


from controlnet_guidance import StableDiffusionControlNet
from stable_diffusion_guidance import StableDiffusion
import unet_generator
from shading_dataset import ColorizationDataset, colorization_collate_fn
from optimizer import Adan
import color_controlnet_trainer
from color_controlnet_trainer import ColorControlNetMultiLayerTrainer
import kornia


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--sd_version",
        type=str,
        default='1.5',
        help="Stable Diffusion version.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        help="Stable Diffusion version.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--min_t_ratio",
        type=float,
        default=0.02,
        help="Minimum T step ratio.",
    )
    parser.add_argument(
        "--max_t_ratio",
        type=float,
        default=0.98,
        help="Minimum T step ratio.",
    )
    parser.add_argument(
        "--multilayer",
        type=bool,
        default=False,
        help="Optimize in latent space or in pixel space?",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=100,
        help="Classifier free guidance scale",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Path to dataset",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        default=None,
        help="Path to dataset",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Path to dataset",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images in the training set",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500,
        help="Number of images in the training set",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5 * 1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--reg_w",
        type=float,
        default=0.,
        help="Learning rate",
    )

    parser.add_argument(
        "--composition_fn",
        type=str,
        default="luminosity",
        choices=["luminosity", "additive"],
        help="Learning rate",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="experiment",
        help="Experiment name",
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="experiment",
        help="Experiment name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A colored professional photo of a strawberry cheesecake on a porcelain plate",
        help="Training prompt",
    )
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="A colored professional photo of a strawberry cheesecake on a porcelain plate",
        help="Training prompt",
    )
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def img_to_gif(img_pattern, out_path, duration=100):
    img_list = sorted(glob.glob(img_pattern))
    pil_img_list = []
    for img in img_list:
        temp = Image.open(img)
        img_copy = temp.copy()
        pil_img_list.append(img_copy)
        temp.close()

    pil_img_list[0].save(out_path, save_all=True, append_images=pil_img_list[1:], duration=duration, loop=0)


def compute_upper_bound_luminosity(img_path_1, img_path_2):
  img_1 = Image.open(img_path_1).resize((512, 512))
  img_2 = Image.open(img_path_2).resize((512, 512))
  img_1_arr = np.array(img_1) / 255.
  img_2_arr = np.array(img_2) / 255.
  luminosity_1 = np.mean(img_1_arr, axis=-1, keepdims=True)
  luminosity_2 = np.mean(img_2_arr, axis=-1, keepdims=True)
  lum_diff = luminosity_2 / (luminosity_1 + 1e-8)
  return np.clip(img_1_arr * lum_diff, 0., 1.)


def additive(base_layer, current_layer, lum_multiplyer=None):
  return base_layer + current_layer

def multiplicative(base_layer, current_layer, lum_multiplyer=None):
  return base_layer + base_layer * current_layer


def compute_luminosity(input_rgb, lum_multiplyer=None):
  if lum_multiplyer is None:
    return torch.mean(input_rgb, axis=1, keepdims=True)
  return torch.sum(input_rgb * lum_multiplyer, axis=1, keepdims=True)

# torch.FloatTensor([0.3, 0.59, 0.11]).reshape([1, 3, 1, 1]).to(device
def multiplicative_luminosity(input_rgb, current_layer, lum_multiplyer=None):
  luminosity = compute_luminosity(input_rgb, lum_multiplyer)
  mult_layer = luminosity * current_layer
  return mult_layer * input_rgb / (luminosity + 1e-9)

def compose_lab_to_rgb_kornia(l_channel_0_1, pred_ab_channels_neg1_1):
    """
    Composes L, A, B channels into an RGB image using kornia.
    Args:
        l_channel_0_1: L channel, normalized to [0, 1]. Shape (B, 1, H, W)
        pred_ab_channels_neg1_1: Predicted A and B channels from U-Net, normalized to [-1, 1]. Shape (B, 2, H, W)
    Returns:
        RGB image tensor, values in [0, 1]. Shape (B, 3, H, W)
    """
    # Scale L from [0, 1] back to [0, 100] for Lab standard
    l_channel_lab_standard = l_channel_0_1 * 100.0

    # Scale predicted A, B from U-Net's [-1, 1] (via tanh) to Lab's typical range.
    # kornia.color.lab_to_rgb expects L in [0, 100], A, B in roughly [-100, 100] to [-128, 127].
    # Scaling tanh output by 110 covers a good range like [-110, 110].
    pred_a_channel_lab_standard = pred_ab_channels_neg1_1[:, 0:1, :, :] * 110.0 
    pred_b_channel_lab_standard = pred_ab_channels_neg1_1[:, 1:2, :, :] * 110.0

    # Concatenate to form Lab image (B, 3, H, W)
    lab_image_tensor = torch.cat([l_channel_lab_standard, pred_a_channel_lab_standard, pred_b_channel_lab_standard], dim=1)

    # Convert Lab to RGB using kornia (differentiable). Output is RGB in [0, 1].
    rgb_image_tensor = kornia.color.lab_to_rgb(lab_image_tensor)
    
    return rgb_image_tensor.clamp(0., 1.) # Ensure output is strictly in [0,1]


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # guidance = StableDiffusionControlNet(
    #     device, False, False, args.sd_version, None, 
    #     min_train_step_scaler=args.min_t_ratio, max_train_step_scaler=args.max_t_ratio,
    #     controlnet_path=args.controlnet_path)
    
    guidance = StableDiffusion(
        device, False, False, args.sd_version, None, 
        min_train_step_scaler=args.min_t_ratio, max_train_step_scaler=args.max_t_ratio,
       )
    
    if args.img_path is not None:
        dataset = ColorizationDataset( # Use new dataset
            instance_data_path=args.img_path,
            device=device,
            H=512, W=512) # Ensure H, W are consistent
    else:
        raise ValueError("img_path must be provided for colorization.")
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=colorization_collate_fn,
        batch_size=args.batch_size,
        num_workers=0)

    model = unet_generator.ColorizationUNet(in_channels=1, out_channels=2)
    
    composition_fn = compose_lab_to_rgb_kornia

    trainer = ColorControlNetMultiLayerTrainer(
            name=args.exp_name,
            model=model,
            guidance=guidance,
            prompt=args.prompt,
            negative_prompt=args.neg_prompt,
            device=device,
            cfg=args.cfg,
            workspace=os.path.join(args.exp_root, args.exp_name),
            fp16=True,
            use_checkpoint="latest",
            iters=args.max_train_steps,
            eval_path=os.path.join(args.exp_root, args.exp_name, "eval"),
            as_latent=False,
            reg_weight=args.reg_w,
            composition_fn=composition_fn
            )
    
    optimizer = lambda model_to_opt: Adan(model_to_opt.parameters(), lr=args.lr, eps=1e-8, weight_decay=2e-5, max_grad_norm=5.0, foreach=False)
    trainer.optimizer = optimizer(model)

    max_epoch = np.ceil(args.max_train_steps / len(train_dataloader)).astype(np.int32)
    trainer.train(train_dataloader, train_dataloader, max_epoch)


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        seed_everything(args.seed)
    main(args)

