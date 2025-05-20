#@title Colorization trainer
import os
import numpy as np
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX
import cv2
import matplotlib.pyplot as plt

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage
import kornia

def convert_to_image(torch_tensor):
  np_image = torch_tensor.detach().cpu().numpy()
  norm_img = (np_image - np_image.min()) / (np_image.max() - np_image.min() + 1e-6)
  img = (norm_img * 255).astype(np.uint8)
  return img


# def additive(base_layer, current_layer, lum_multiplyer=None):
#   return base_layer + current_layer

# def multiplicative(base_layer, current_layer, lum_multiplyer=None):
#   return base_layer * current_layer


# def devisive(base_layer, current_layer, lum_multiplyer=None):
#   return base_layer / (current_layer + 1e-6)


# def mult_and_divisive(base_layer, current_mult_layer, current_div_layer, lum_multiplyer=None):
#   return (base_layer * current_mult_layer) / (current_div_layer + 1e-6)

# # torch.FloatTensor([0.3, 0.59, 0.11]).reshape([1, 3, 1, 1]).to(device
# def multiplicative_luminosity(input_rgb, current_layer, lum_multiplyer=None):
#   luminosity = compute_luminosity(input_rgb, lum_multiplyer)
#   mult_layer = luminosity * current_layer
#   return mult_layer * input_rgb / (luminosity + 1e-9)

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

class ColorControlNetMultiLayerTrainer(object):
    def __init__(self,
                 name, # name of this experiment
                 model, # network
                 guidance, # guidance network
                 prompt="",
                 negative_prompt="",
                 iters=500,
                 cfg=100,
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 fp16=False, # amp optimize level
                 eval_interval=50, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 use_tensorboardX=True,
                 warmup_iters=200,
                 eval_path="",  # intermediate results will be saved here,
                 as_latent=True,
                 reg_weight=0.,
                 composition_fn=None,
                 ):

        self.name = name
        self.warmup_iters = warmup_iters
        self.iters = iters
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.log_ptr = None
        self.use_checkpoint = use_checkpoint
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.use_tensorboardX = use_tensorboardX
        self.cfg = cfg
        self.reg_weight = reg_weight
        self.as_latent = True
        self.eval_path = eval_path if len(eval_path) > 0 else None

        self.lum_layer = torch.FloatTensor([0.3, 0.59, 0.11]).reshape([1, 3, 1, 1]).to(device)
        os.makedirs(self.eval_path, exist_ok=True)
        if composition_fn is None:
          self.composition_fn = compose_lab_to_rgb_kornia
        else:
            self.composition_fn = composition_fn

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        # guide model
        self.guidance = guidance

        # text prompt
        if self.guidance is not None:

            for p in self.guidance.parameters():
                p.requires_grad = False

            self.prepare_text_embeddings()

        else:
            self.text_z = None

        # try out torch 2.0
        if torch.__version__[0] == '2':
            self.model = torch.compile(self.model)
            self.guidance = torch.compile(self.guidance)

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.prompt is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        # construct dir-encoded text
        text = f"{self.prompt}"
        negative_text = f"{self.negative_prompt}"
        self.text_z = self.guidance.get_text_embeds([text], [negative_text])

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    def train_step(self, data, as_latent=False):
        print(f"[INFO] as_latent: {as_latent} in color controlnet trainer")
        # input_latents = self.guidance.encode_imgs(data["input_pixels"])
        l_channel_input = data['l_channel_unet_input']
        l_channel_compose = data['l_channel_for_composition']
        control_image_cond = data['control_image_for_controlnet'] # L replicated to 3 channels        
        pred_ab_channels = self.model(l_channel_input) # Output is tanh scaled, shape (B, 2, H, W)
        pred_rgb = self.composition_fn(l_channel_compose, pred_ab_channels)
        
        loss = self.guidance.train_step(self.text_z, pred_rgb, 
                                         # This is the ControlNet condition
                                        guidance_scale=self.cfg, as_latent=False) 
        
        # Optional: Regularization on A,B channels if needed (e.g., to prevent extreme values)
        # if self.reg_weight > 0:
        #     reg_loss = torch.mean(pred_ab_channels**2) 
        #     loss = loss + self.reg_weight * reg_loss
            
        return pred_rgb, loss # Return pred_rgb for potential logging during training



    def post_train_step(self):

        if self.opt.backbone == 'grid' and self.opt.lambda_tv > 0:
            lambda_tv = min(1.0, self.global_step / 1000) * self.opt.lambda_tv
            # unscale grad before modifying it!
            # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
            self.scaler.unscale_(self.optimizer)
            self.model.encoder.grad_total_variation(lambda_tv, None, self.model.bound)

    def eval_step(self, data): # data is from train_dataloader in this setup
        l_channel_input = data['l_channel_unet_input']
        l_channel_compose = data['l_channel_for_composition']
        loss = 0
        with torch.no_grad():
            pred_ab_channels = self.model(l_channel_input)
            res_rgb = self.composition_fn(l_channel_compose, pred_ab_channels)
            
            # Calculate loss for eval if needed (optional, but good for tracking)
            # loss = self.guidance.train_step(self.text_embeds_map['positive'], res_rgb, 
            #                                 guidance_scale=self.cfg, as_latent=self.as_latent, grad_scale=self.grad_scale)
                
                
        loss = torch.mean(torch.abs(res_rgb - data['l_channel_unet_input']), dtype=res_rgb.dtype)

        
        # Return components for logging
        # Instead of pred_mult_layer, pred_div_layer, return pred_ab_channels
        return pred_ab_channels, res_rgb, loss

    def test_step(self, data, bg_color=None, perturb=False):
        pred_mult_layer, pred_div_layer = self.model(data["input_pixels"])
        res_rgb = self.composition_fn(data["input_pixels"], pred_mult_layer, pred_div_layer)
        return res_rgb
    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

        assert self.text_z is not None, 'Training must provide a text prompt!'

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            self.train_one_epoch(train_loader, max_epochs)

            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)


        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                # data = data.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, _ = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8, macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8, macro_block_size=1)

        self.log(f"==> Finished Test.")


    def train_one_epoch(self, loader, max_epochs):
        self.log(f"==> Start Training {self.workspace} Epoch {self.epoch}/{max_epochs}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
        self.guidance.requires_grad = False

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0
        # self.prev_res = torch.zeros_like(next(iter(loader))["input_pixels"]).to(self.device)
        for data in loader:

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()
            # data = data.to(self.device)
            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, loss = self.train_step(data, as_latent=self.as_latent)
               
            # self.optimizer.step() # Gardient Descent
            self.scaler.scale(loss).backward()
            # self.post_train_step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}/{max_epochs}.")

    def log_eval_imgs(self, data, eval_results, step=0):
        pred_ab_channels, composed_rgb_img_tensor, _ = eval_results # From eval_step
        
        # Get first item in batch for saving
        l_input_to_save = data['l_channel_unet_input'][0].cpu().permute(1,2,0).numpy().squeeze() # (H,W)
        pred_a_to_save = ((pred_ab_channels[0, 0, :, :].cpu().numpy() + 1.0) / 2.0) # Scale [-1,1] to [0,1]
        pred_b_to_save = ((pred_ab_channels[0, 1, :, :].cpu().numpy() + 1.0) / 2.0) # Scale [-1,1] to [0,1]
        final_rgb_to_save = composed_rgb_img_tensor[0] # Tensor (3,H,W) in [0,1]

        plt.imsave(os.path.join(self.eval_path, f"epoch_{self.epoch:05}_input_L.png"), l_input_to_save, cmap='gray')
        plt.imsave(os.path.join(self.eval_path, f"epoch_{self.epoch:05}_pred_A_norm.png"), pred_a_to_save, cmap='viridis')
        plt.imsave(os.path.join(self.eval_path, f"epoch_{self.epoch:05}_pred_B_norm.png"), pred_b_to_save, cmap='viridis')
        
        # Use convert_to_image for the final RGB (instance-wise normalization for visualization)
        plt.imsave(os.path.join(self.eval_path, f"epoch_{self.epoch:05}_composed_RGB_evalnorm.png"), 
                   convert_to_image(final_rgb_to_save.permute((1, 2, 0)))) # convert_to_image expects (H,W,C)

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    eval_results = self.eval_step(data)
                    # print(preds.min(), preds.max(), loss)
                    loss = eval_results[2] 
                # all_gather/reduce the statistics (NCCL only support all_*)
                # if self.world_size > 1:
                #     dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                #     loss = loss / self.world_size

                #     preds_list = [torch.zeros_like(res_rgb).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                #     dist.all_gather(preds_list, preds)
                #     preds = torch.cat(preds_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    # save image
                    self.log_eval_imgs(data, eval_results, step=self.local_step)

                    # Save the "validation" style image (clipped, not instance-normalized)
                     # This was the _rgb.png in the validation folder
                    current_composed_rgb_for_validation = eval_results[1] # res_rgb
                    save_path_val = os.path.join(self.workspace, 'validation', f'{self.name}_ep{self.epoch:04d}_rgb.png')
                    os.makedirs(os.path.dirname(save_path_val), exist_ok=True)
                     
                    pred_val_np = current_composed_rgb_for_validation[0].detach().cpu().numpy() # (3,H,W)
                    pred_val_np = np.clip(pred_val_np, 0., 1.)
                    pred_val_np_uint8 = (pred_val_np * 255).astype(np.uint8)
                     # Transpose to (H,W,3) for cv2
                    cv2.imwrite(save_path_val, cv2.cvtColor(pred_val_np_uint8.transpose((1, 2, 0)), cv2.COLOR_RGB2BGR))


        average_loss = total_loss / self.local_step
        self.log(f"++ Finished evaluation at epoch {self.epoch}. Average loss: {average_loss}")

        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")
        return average_loss

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                # always save best since loss cannot reflect performance.
                if True:
                    # self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    # self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")