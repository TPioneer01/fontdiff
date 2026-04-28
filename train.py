import os
import math
import time
import logging
import csv
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (FontDiffuserModel,
                 FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 ContentPerceptualLoss,
                 EdgeConsistencyLoss,
                 build_unet,
                 build_style_encoder,
                 build_content_encoder,
                 build_ddpm_scheduler)
from src import build_scr
from utils import (save_args_to_yaml,
                   x0_from_epsilon, 
                   reNormalize_img, 
                   normalize_mean_std,
                   save_inference_batch_results)
from sample import inference_on_dataset_samples


logger = get_logger(__name__)

def get_args():
    parser = get_parser()
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def main():

    args = get_args()

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # Ser training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    noise_scheduler = build_ddpm_scheduler(args)
    if args.phase_2:
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth"))
        content_encoder.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth"))

    model = FontDiffuserModel(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        edge_fusion_scale=args.edge_fusion_scale)

    if args.phase_2:
        edge_adapter_path = f"{args.phase_1_ckpt_dir}/edge_adapter.pth"
        if os.path.exists(edge_adapter_path):
            try:
                edge_state = torch.load(edge_adapter_path, map_location="cpu")
                if "content_adapter" in edge_state and "style_adapter" in edge_state and "fusion_scale" in edge_state:
                    model.edge_adapter_content.load_state_dict(edge_state["content_adapter"])
                    model.edge_adapter_style.load_state_dict(edge_state["style_adapter"])
                    # Safely update edge_fusion_scale using the property setter
                    try:
                        model.edge_fusion_scale = edge_state["fusion_scale"]
                    except Exception as e:
                        print(f"[Warning] Could not update edge_fusion_scale: {e}")
                else:
                    model.load_state_dict(edge_state, strict=False)
                print("Loaded edge adapter from phase-1 checkpoint.")
            except Exception as e:
                print(f"[Warning] Failed to load edge adapter: {e}")
                import traceback
                traceback.print_exc()

    # Build content perceptaual Loss
    perceptual_loss = ContentPerceptualLoss()
    edge_loss_fn = EdgeConsistencyLoss()

    # Load SCR module for supervision
    if args.phase_2:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)

    # Load the datasets
    content_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    style_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    target_transforms = transforms.Compose(
        [transforms.Resize((args.resolution, args.resolution), 
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    train_font_dataset = FontDataset(
        args=args,
        phase='train', 
        transforms=[
            content_transforms, 
            style_transforms, 
            target_transforms],
        scr=args.phase_2)
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=CollateFN())
    
    # Build optimizer and learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,)

    # Accelerate preparation
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)
    ## move scr module to the target deivces
    if args.phase_2:
        scr = scr.to(accelerator.device)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(args=args, output_file=f"{args.output_dir}/{args.experience_name}_config.yaml")

    # ============ Initialize loss history tracking for detailed logging ============
    loss_history = []  # List of dicts: {'step': int, 'diff_loss': float, ...}
    loss_csv_path = f"{args.output_dir}/loss_breakdown.csv"
    
    # ============ Initialize inference pipeline for training-time monitoring ============
    inference_pipe = None
    if args.inference_during_training and accelerator.is_main_process:
        try:
            # Initialize inference components
            unet_temp = build_unet(args=args)
            style_encoder_temp = build_style_encoder(args=args)
            content_encoder_temp = build_content_encoder(args=args)
            
            # Create DPM model (will be updated with trained weights later)
            model_dpm = FontDiffuserModelDPM(
                unet=unet_temp,
                style_encoder=style_encoder_temp,
                content_encoder=content_encoder_temp,
                edge_fusion_scale=args.edge_fusion_scale
            )
            
            # Load DDPM scheduler
            noise_scheduler_infer = build_ddpm_scheduler(args)
            
            # Create inference pipeline
            inference_pipe = FontDiffuserDPMPipeline(
                model=model_dpm,
                ddpm_train_scheduler=noise_scheduler_infer,
                model_type=args.model_type,
                guidance_type=args.guidance_type,
                guidance_scale=args.guidance_scale,
            )
            print("[Inference Pipeline] Initialized for training-time monitoring")
        except Exception as e:
            print(f"[Warning] Failed to initialize inference pipeline: {e}")
            import traceback
            traceback.print_exc()
            inference_pipe = None
            args.inference_during_training = False

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            content_edges = samples["content_edge"]
            style_images = samples["style_image"]
            style_edges = samples["style_edge"]
            target_images = samples["target_image"]
            target_edges = samples["target_edge"]
            nonorm_target_images = samples["nonorm_target_image"]
            
            with accelerator.accumulate(model):
                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=target_images.device)
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_target_images = noise_scheduler.add_noise(target_images, noise, timesteps)

                # Classifier-free training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value==1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1
                        content_edges[i, :, :, :] = 0
                        style_edges[i, :, :, :] = 0

                # Predict the noise residual and compute loss
                noise_pred, offset_out_sum = model(
                    x_t=noisy_target_images, 
                    timesteps=timesteps, 
                    style_images=style_images,
                    content_images=content_images,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    content_edges=content_edges if args.use_edge_condition else None,
                    style_edges=style_edges if args.use_edge_condition else None)
                diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                offset_loss = offset_out_sum / 2
                
                # output processing for content perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps)
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device)
                edge_loss = edge_loss_fn(pred_original_sample, nonorm_target_images, target_edge_maps=target_edges)
                
                # ========== Compute total loss and track components ==========
                loss = diff_loss + \
                        args.perceptual_coefficient * percep_loss + \
                            args.offset_coefficient * offset_loss + \
                            args.edge_coefficient * edge_loss
                
                # Track loss components for detailed logging
                loss_components = {
                    'diff_loss': diff_loss.detach().item(),
                    'percep_loss': percep_loss.detach().item(),
                    'offset_loss': offset_loss.detach().item(),
                    'edge_loss': edge_loss.detach().item(),
                }
                
                sc_loss = None
                if args.phase_2:
                    neg_images = samples["neg_images"]
                    # sc loss
                    sample_style_embeddings, pos_style_embeddings, neg_style_embeddings = scr(
                        pred_original_sample_norm, 
                        target_images, 
                        neg_images, 
                        nce_layers=args.nce_layers)
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings)
                    loss += args.sc_coefficient * sc_loss
                    loss_components['sc_loss'] = sc_loss.detach().item()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.ckpt_interval == 0:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
                        torch.save(model.style_encoder.state_dict(), f"{save_dir}/style_encoder.pth")
                        torch.save(model.content_encoder.state_dict(), f"{save_dir}/content_encoder.pth")
                        torch.save(
                            {
                                "content_adapter": model.edge_adapter_content.state_dict(),
                                "style_adapter": model.edge_adapter_style.state_dict(),
                                "fusion_scale": model.edge_fusion_scale.detach().cpu(),
                            },
                            f"{save_dir}/edge_adapter.pth",
                        )
                        torch.save(model, f"{save_dir}/total_model.pth")
                        logging.info(f"[{time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}] Save the checkpoint on global step {global_step}")
                        print("Save the checkpoint on global step {}".format(global_step))
                        
                        # ========== Perform inference on dataset samples ==========
                        if args.inference_during_training and inference_pipe is not None:
                            try:
                                print(f"[Checkpoint {global_step}] Performing inference on dataset samples...")
                                
                                # Update inference pipeline with current model weights
                                # Load model components
                                inference_pipe.model.unet.load_state_dict(model.unet.state_dict())
                                inference_pipe.model.style_encoder.load_state_dict(model.style_encoder.state_dict())
                                inference_pipe.model.content_encoder.load_state_dict(model.content_encoder.state_dict())
                                inference_pipe.model.edge_adapter_content.load_state_dict(model.edge_adapter_content.state_dict())
                                inference_pipe.model.edge_adapter_style.load_state_dict(model.edge_adapter_style.state_dict())
                                
                                # Safely update edge_fusion_scale using property setter
                                try:
                                    inference_pipe.model.edge_fusion_scale = model.edge_fusion_scale.detach()
                                except Exception as e:
                                    print(f"[Warning] Could not update edge_fusion_scale: {e}")
                                
                                inference_pipe.model.to(accelerator.device)
                                
                                # Run inference
                                gen_images, cnt_images, sty_images, sample_ids = inference_on_dataset_samples(
                                    args=args,
                                    pipe=inference_pipe,
                                    train_dataset=train_font_dataset,
                                    device=accelerator.device,
                                    num_samples=args.num_inference_samples,
                                    seed=args.inference_seed + global_step  # Vary seed per checkpoint
                                )
                                
                                # Save inference results
                                inference_save_dir = f"{save_dir}/inference_samples"
                                save_inference_batch_results(
                                    save_dir=inference_save_dir,
                                    generated_images=gen_images,
                                    content_images_pil=cnt_images,
                                    style_images_pil=sty_images,
                                    sample_indices=sample_ids,
                                    resolution=args.resolution
                                )
                                print(f"[Checkpoint {global_step}] Inference results saved to {inference_save_dir}")
                                logging.info(f"Inference samples saved at checkpoint {global_step}")
                                
                            except Exception as e:
                                print(f"[Checkpoint {global_step}] Inference failed: {e}")
                                logging.warning(f"Inference error at checkpoint {global_step}: {e}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if global_step % args.log_interval == 0:
                # ========== Detailed loss component logging ==========
                log_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] " \
                          f"Global Step {global_step} | Total Loss: {loss.detach().item():.6f} | " \
                          f"diff_loss: {loss_components['diff_loss']:.6f} | " \
                          f"percep_loss: {loss_components['percep_loss']:.6f} | " \
                          f"offset_loss: {loss_components['offset_loss']:.6f} | " \
                          f"edge_loss: {loss_components['edge_loss']:.6f}"
                if args.phase_2 and 'sc_loss' in loss_components:
                    log_msg += f" | sc_loss: {loss_components['sc_loss']:.6f}"
                
                logging.info(log_msg)
                print(log_msg)
                
                # ========== Store loss components for CSV export ==========
                log_entry = {'step': global_step}
                log_entry.update(loss_components)
                loss_history.append(log_entry)
                
                # ========== Log to TensorBoard ==========
                accelerator.log(
                    {
                        "train_loss": loss.detach().item(),
                        "diff_loss": loss_components['diff_loss'],
                        "percep_loss": loss_components['percep_loss'],
                        "offset_loss": loss_components['offset_loss'],
                        "edge_loss": loss_components['edge_loss'],
                        **({"sc_loss": loss_components['sc_loss']} if 'sc_loss' in loss_components else {})
                    },
                    step=global_step
                )
            progress_bar.set_postfix(**logs)
            
            # Quit
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()
    
    # ========== Export loss history to CSV ==========
    if accelerator.is_main_process and loss_history:
        try:
            import pandas as pd
            df = pd.DataFrame(loss_history)
            df.to_csv(loss_csv_path, index=False)
            print(f"\n[Training Complete] Loss breakdown saved to: {loss_csv_path}")
            logging.info(f"Loss history exported to {loss_csv_path}")
        except ImportError:
            # Fallback to csv module if pandas not available
            print(f"\n[Training Complete] Exporting loss history using csv module...")
            if loss_history:
                fieldnames = list(loss_history[0].keys())
                with open(loss_csv_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(loss_history)
                print(f"Loss breakdown saved to: {loss_csv_path}")
                logging.info(f"Loss history exported to {loss_csv_path}")

if __name__ == "__main__":
    main()
