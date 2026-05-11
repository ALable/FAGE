"""PersonaGazeNet — Training script (3-phase curriculum).

Usage:
    accelerate launch --num_processes 1 train_persona_gaze.py \
        --config configs/training/persona_gaze.yaml

The ``phase`` field in the config selects which modules are trainable:
    phase1 → encoder + decoder + gaze_mlp (no refs, gaze-only)
    phase2 → + ref_tokenizer + RFA gates
    phase3 → + HyperLoRA

For curriculum learning, train phase1 first, then resume from its checkpoint
into phase2 (update cfg.phase + cfg.resume_from_checkpoint), and so on.

This script intentionally mirrors ``train_facetoeyenet.py`` for consistency
with the rest of the FAGE repo (Accelerate, TensorBoard, paste_eyes, etc.).
"""
import argparse
import logging
import math
import os
import shutil
import warnings
from datetime import datetime, timedelta

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, LoggerType
from omegaconf import OmegaConf
from piq import psnr as piq_psnr
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from dataset.multi_ref_dataset import MultiRefHDFDataset
from gaze_estimation.baseline_resnet import GazeHeadResNet
from loss.basic_loss import discriminator_loss, gaze_angular_loss, generator_loss
from models.gazenet import MLPNetwork
from models.persona_gaze import PersonaGazeNet
from utils.training_utils import initialize_loss_functions

logger = get_logger(__name__, log_level="INFO")
warnings.filterwarnings("ignore")


# ── Utilities ─────────────────────────────────────────────────────────────────


def seed_everything(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg) -> PersonaGazeNet:
    params = OmegaConf.to_container(cfg.pgn_params, resolve=True)
    return PersonaGazeNet(**params)


def build_gaze_mlp(cfg) -> MLPNetwork:
    g = cfg.model_params.gazenet_params
    return MLPNetwork(
        num_in=g.num_in,
        num_hidden=g.num_hidden,
        num_out=cfg.pgn_params.get("gaze_dim", 64),
        num_layers=g.num_layers,
        cross_condition=g.get("cross_condition", False),
    )


def build_lr_scheduler(optimizer, cfg):
    warmup = cfg.solver.lr_warmup_steps
    total = cfg.solver.max_train_steps

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Losses specific to PGN ────────────────────────────────────────────────────


def whitening_loss(head_emb: torch.Tensor, gaze_emb: torch.Tensor) -> torch.Tensor:
    """Encourage head_emb and gaze_emb to live in (approximately) decorrelated
    subspaces — keeps gaze geometry from leaking into identity channels."""
    B = head_emb.shape[0]
    z = torch.cat([head_emb, gaze_emb], dim=0)        # [2B, D]
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / max(1, z.shape[0] - 1)
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return F.mse_loss(cov, eye, reduction="mean")


def cycle_loss(model, gaze_mlp,
               first_pred: torch.Tensor,
               source_eye: torch.Tensor,
               source_head: torch.Tensor,
               source_gaze: torch.Tensor,
               ref_eyes: torch.Tensor) -> torch.Tensor:
    """Identity-anchored cycle: feed ``first_pred`` as a fresh reference and ask
    the model to reconstruct ``source_eye`` under the *source* gaze.

    Uses ``first_pred.detach()`` to avoid double backward through the same path.
    """
    fp = first_pred.detach()
    # New reference set = original refs + the generated prediction (encourages
    # the predicted appearance to round-trip).
    cycle_refs = torch.cat([ref_eyes, fp.unsqueeze(1)], dim=1)
    h_emb, g_emb = gaze_mlp(source_head, source_gaze)
    gaze_cond = torch.cat([h_emb.unsqueeze(1), g_emb.unsqueeze(1)], dim=1)
    cycle_pred = model(source_eye, gaze_cond, ref_eyes=cycle_refs)
    return F.l1_loss(cycle_pred, source_eye)


# ── Visualisation ─────────────────────────────────────────────────────────────


def _tb_writer(accelerator):
    for tr in accelerator.trackers:
        if tr.name == "tensorboard":
            return tr.writer
    return None


def _denorm(t):
    return (t * 0.5 + 0.5).clamp(0.0, 1.0)


def log_vis(accelerator, step, source_eye, gen, target, refs, n=4, prefix="train"):
    writer = _tb_writer(accelerator)
    if writer is None:
        return
    n = min(n, source_eye.shape[0])

    def grid(t):
        return make_grid(_denorm(t[:n].detach().float().cpu()), nrow=n, padding=2)

    eye_rows = torch.cat([grid(source_eye), grid(gen), grid(target)], dim=1)
    writer.add_image(f"{prefix}/eye_src_gen_gt", eye_rows, step)

    if refs is not None:
        # refs shape [B, K, 3, H, W] — concat first 2 refs along width
        K = min(refs.shape[1], 4)
        ref_grid = make_grid(
            _denorm(refs[:n, :K].reshape(n * K, *refs.shape[2:]).detach().float().cpu()),
            nrow=K, padding=2,
        )
        writer.add_image(f"{prefix}/refs", ref_grid, step)


# ── Main ─────────────────────────────────────────────────────────────────────


def main(cfg, config_path=None):
    save_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    if config_path and os.path.exists(config_path):
        shutil.copy2(config_path, os.path.join(save_dir, "config.yaml"))
    else:
        OmegaConf.save(cfg, os.path.join(save_dir, "config.yaml"))

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        log_with=["tensorboard", LoggerType.TENSORBOARD],
        mixed_precision=cfg.solver.mixed_precision,
        project_dir=os.path.join(save_dir, "tensorboard"),
        kwargs_handlers=[
            DistributedDataParallelKwargs(),
            InitProcessGroupKwargs(timeout=timedelta(seconds=5400)),
        ],
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)

    if cfg.seed is not None:
        seed_everything(cfg.seed + accelerator.process_index)

    weight_dtype = torch.float32
    phase = cfg.get("phase", "phase1")
    use_refs = phase in ("phase2", "phase3")

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(cfg)
    gaze_mlp = build_gaze_mlp(cfg)
    model.set_phase(phase)
    total = sum(p.numel() for p in model.parameters())
    train_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"PGN [{phase}]: {total/1e6:.2f}M total, "
                f"{train_count/1e6:.2f}M trainable")

    # ── Optimizer / scheduler ────────────────────────────────────────────────
    opt_params = [p for p in model.parameters() if p.requires_grad] + list(gaze_mlp.parameters())
    optimizer = torch.optim.AdamW(
        opt_params,
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = build_lr_scheduler(optimizer, cfg)

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds_kwargs = dict(
        split_ratio=cfg.data.split_ratio,
        seed=cfg.seed,
        frame_offset_range=cfg.data.get("frame_offset_range", 12),
        eye_crop_size=cfg.data.get("eye_crop_size", [64, 64]),
        eye_expand_ratio=cfg.data.get("eye_expand_ratio", 1.5),
        input_eye_crop_size=cfg.data.get("input_eye_crop_size", None),
        input_eye_expand_ratio=cfg.data.get("input_eye_expand_ratio", None),
        max_head_diff=cfg.data.get("max_head_diff", None),
        max_target_retries=cfg.data.get("max_target_retries", 10),
        ref_K=cfg.data.get("ref_K", 4),
        ref_K_choices=cfg.data.get("ref_K_choices", None) if phase == "phase3" else None,
        max_head_diff_ref=cfg.data.get("max_head_diff_ref", None),
    )

    train_ds = MultiRefHDFDataset(cfg.data.hdf_path, split="train", **ds_kwargs)
    val_ds   = MultiRefHDFDataset(cfg.data.hdf_path, split="val",   **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.train_bs, shuffle=True,
                              num_workers=cfg.data.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.train_bs, shuffle=False,
                            num_workers=cfg.data.num_workers, pin_memory=True)

    model, gaze_mlp, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        model, gaze_mlp, optimizer, lr_scheduler, train_loader, val_loader,
    )

    # ── Loss helpers (reuse FAGE infra) ──────────────────────────────────────
    loss_dict = initialize_loss_functions(cfg, accelerator, cfg.solver.max_train_steps)
    if "eye_discriminator" in loss_dict:
        loss_dict["eye_discriminator"], loss_dict["eye_optimizer_D"] = accelerator.prepare(
            loss_dict["eye_discriminator"], loss_dict["eye_optimizer_D"]
        )

    lpips_fn = lpips.LPIPS(net="alex").to(accelerator.device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False

    gaze_estimator = GazeHeadResNet(pretrained=False).to(accelerator.device)
    if os.path.exists(cfg.pretrained.gaze_eval_checkpoint_path):
        ckpt = torch.load(cfg.pretrained.gaze_eval_checkpoint_path, map_location=accelerator.device)
        gaze_estimator.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded gaze estimator from {cfg.pretrained.gaze_eval_checkpoint_path}")
    gaze_estimator.eval()
    for p in gaze_estimator.parameters():
        p.requires_grad = False

    gaze_eval_trans = T.Compose([
        T.Resize((224, 224)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ── Setup ────────────────────────────────────────────────────────────────
    num_steps_per_epoch = math.ceil(len(train_loader) / cfg.solver.gradient_accumulation_steps)
    num_epochs = math.ceil(cfg.solver.max_train_steps / num_steps_per_epoch)
    if accelerator.is_main_process:
        run_name = datetime.now().strftime("%Y%m%d-%H%M") + f"_{phase}"
        accelerator.init_trackers(cfg.exp_name, init_kwargs={"mlflow": {"run_name": run_name}})

    logger.info(f"PGN training — phase={phase}, "
                f"epochs={num_epochs}, max_steps={cfg.solver.max_train_steps}, "
                f"train_samples={len(train_ds)}")

    global_step = 0

    # Resume
    resume = cfg.get("resume_from_checkpoint", False)
    if resume and isinstance(resume, str) and os.path.exists(resume):
        logger.info(f"Resuming from {resume}")
        ckpt = torch.load(resume, map_location="cpu")
        global_step = ckpt.get("global_step", 0)
        strict = cfg.get("resume_strict", False)
        accelerator.unwrap_model(model).load_state_dict(ckpt["model_state_dict"], strict=strict)
        accelerator.unwrap_model(gaze_mlp).load_state_dict(ckpt["gaze_mlp_state_dict"], strict=strict)
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                logger.warning(f"Optimizer reset (phase change): {e}")
        if "lr_scheduler_state_dict" in ckpt and cfg.get("resume_lr", True):
            try:
                lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
            except Exception as e:
                logger.warning(f"LR scheduler reset: {e}")
        # Re-apply phase mask after loading
        accelerator.unwrap_model(model).set_phase(phase)

    progress = tqdm(range(cfg.solver.max_train_steps), initial=global_step,
                    disable=not accelerator.is_local_main_process)
    best_score = float("inf")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        model.train()
        gaze_mlp.train()

        for step, batch in enumerate(train_loader):
            if global_step >= cfg.solver.max_train_steps:
                break

            with accelerator.accumulate(model, gaze_mlp):
                source_eye  = batch["source_input_eye_crops"].to(weight_dtype)
                target_eye  = batch["target_eye_crops"].to(weight_dtype)
                target_gaze = batch["target_gaze"].to(weight_dtype)
                target_head = batch["target_head"].to(weight_dtype)
                source_gaze = batch["source_gaze"].to(weight_dtype)
                source_head = batch["source_head"].to(weight_dtype)
                ref_eyes    = batch["ref_eye_crops"].to(weight_dtype) if use_refs else None

                # Note: input crops are 80×160, model wants 64×128. We resize on the
                # fly with bicubic to match the GT supervision resolution.
                if source_eye.shape[-2:] != target_eye.shape[-2:]:
                    source_eye = F.interpolate(
                        source_eye, size=target_eye.shape[-2:],
                        mode="bicubic", align_corners=False,
                    )
                if ref_eyes is not None and ref_eyes.shape[-2:] != target_eye.shape[-2:]:
                    B, K = ref_eyes.shape[:2]
                    ref_eyes = F.interpolate(
                        ref_eyes.reshape(B * K, *ref_eyes.shape[2:]),
                        size=target_eye.shape[-2:], mode="bicubic", align_corners=False,
                    ).reshape(B, K, *target_eye.shape[-3:])

                # Gaze embedding
                head_emb, gaze_emb = gaze_mlp(target_head, target_gaze)
                gaze_cond = torch.cat([head_emb.unsqueeze(1), gaze_emb.unsqueeze(1)], dim=1)

                gen = model(source_eye, gaze_cond, ref_eyes=ref_eyes)

                # ── Losses ───────────────────────────────────────────────────
                l1 = F.l1_loss(gen, target_eye)
                loss = cfg.loss_params.l1_loss * l1

                if cfg.loss_params.get("whitening_loss", 0) > 0:
                    w_loss = whitening_loss(head_emb, gaze_emb)
                    loss = loss + cfg.loss_params.whitening_loss * w_loss
                else:
                    w_loss = None

                gaze_perc = None
                if (cfg.loss_params.gaze_perceptual_loss > 0 and
                        global_step >= cfg.loss_params.get("start_gaze_perceptual", 0)):
                    # Use eye crop (left half) for gaze perceptual loss
                    half = gen.shape[3] // 2
                    gen_left = (gen[:, :, :, :half].float() + 1) / 2
                    gen_left = gaze_eval_trans(gen_left)
                    pred_g, _ = gaze_estimator(gen_left)
                    gaze_perc = gaze_angular_loss(target_gaze.float(), pred_g)
                    loss = loss + cfg.loss_params.gaze_perceptual_loss * gaze_perc

                cycle_l = None
                if (use_refs and cfg.loss_params.get("cycle_loss", 0) > 0 and
                        global_step >= cfg.loss_params.get("start_cycle_loss", 0)):
                    cycle_l = cycle_loss(
                        accelerator.unwrap_model(model),
                        accelerator.unwrap_model(gaze_mlp),
                        gen, source_eye, source_head, source_gaze, ref_eyes,
                    )
                    loss = loss + cfg.loss_params.cycle_loss * cycle_l

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(opt_params, cfg.solver.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress.update(1)

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % 100 == 0 and accelerator.is_main_process:
                log = {
                    "train/l1": l1.item(),
                    "train/total": loss.item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                }
                if w_loss is not None:
                    log["train/whitening"] = w_loss.item()
                if gaze_perc is not None:
                    log["train/gaze_perc"] = gaze_perc.item()
                if cycle_l is not None:
                    log["train/cycle"] = cycle_l.item()
                accelerator.log(log, step=global_step)

            # ── Validation ────────────────────────────────────────────────────
            if global_step % cfg.val_freq == 0 and accelerator.is_main_process:
                model.eval()
                gaze_mlp.eval()
                v_l1 = v_lpips = v_psnr = v_gaze = 0.0
                n_val = 0
                vis = None
                with torch.no_grad():
                    for vi, vb in enumerate(val_loader):
                        if vi >= cfg.val_batch_limit:
                            break
                        s_eye = vb["source_input_eye_crops"].to(weight_dtype)
                        t_eye = vb["target_eye_crops"].to(weight_dtype)
                        tg = vb["target_gaze"].to(weight_dtype)
                        th = vb["target_head"].to(weight_dtype)
                        rf = vb["ref_eye_crops"].to(weight_dtype) if use_refs else None
                        if s_eye.shape[-2:] != t_eye.shape[-2:]:
                            s_eye = F.interpolate(s_eye, size=t_eye.shape[-2:],
                                                  mode="bicubic", align_corners=False)
                        if rf is not None and rf.shape[-2:] != t_eye.shape[-2:]:
                            B, K = rf.shape[:2]
                            rf = F.interpolate(
                                rf.reshape(B * K, *rf.shape[2:]),
                                size=t_eye.shape[-2:], mode="bicubic", align_corners=False,
                            ).reshape(B, K, *t_eye.shape[-3:])

                        he, ge = gaze_mlp(th, tg)
                        gc = torch.cat([he.unsqueeze(1), ge.unsqueeze(1)], dim=1)
                        g = model(s_eye, gc, ref_eyes=rf)
                        v_l1    += F.l1_loss(g, t_eye).item()
                        v_lpips += lpips_fn(g, t_eye).mean().item()
                        v_psnr  += piq_psnr((g.float() + 1).clamp(0, 2),
                                            (t_eye.float() + 1).clamp(0, 2), data_range=2.0).item()
                        half = g.shape[3] // 2
                        gl = (g[:, :, :, :half].float() + 1) / 2
                        gl = gaze_eval_trans(gl)
                        pg, _ = gaze_estimator(gl)
                        v_gaze += gaze_angular_loss(tg.float(), pg).item()
                        n_val += 1
                        if vis is None:
                            vis = (s_eye, g, t_eye, rf)

                if n_val > 0:
                    avg_l1, avg_lpips, avg_psnr, avg_gaze = (
                        v_l1 / n_val, v_lpips / n_val, v_psnr / n_val, v_gaze / n_val,
                    )
                    score = (avg_l1 / 0.05 + avg_lpips / 0.1 +
                             1.0 / max(avg_psnr, 1.0) / 0.03 + avg_gaze / 5.0)
                    accelerator.log({
                        "val/l1": avg_l1, "val/lpips": avg_lpips,
                        "val/psnr": avg_psnr, "val/gaze_error": avg_gaze,
                        "val/composite": score,
                    }, step=global_step)
                    logger.info(f"Val [{global_step}] l1={avg_l1:.4f} "
                                f"lpips={avg_lpips:.4f} psnr={avg_psnr:.2f} "
                                f"gaze={avg_gaze:.4f}°  score={score:.3f}")

                    if score < best_score:
                        best_score = score
                        best_path = os.path.join(save_dir, "best_model.pth")
                        torch.save({
                            "global_step": global_step,
                            "best_score": best_score,
                            "phase": phase,
                            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                            "gaze_mlp_state_dict": accelerator.unwrap_model(gaze_mlp).state_dict(),
                        }, best_path)
                        logger.info(f"New best (score={score:.3f}) → {best_path}")

                    if vis is not None:
                        log_vis(accelerator, global_step, *vis[:3], vis[3], prefix="val")
                log_vis(accelerator, global_step,
                        source_eye, gen, target_eye, ref_eyes, prefix="train")
                model.train()
                gaze_mlp.train()

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                ckpt_path = os.path.join(save_dir, f"checkpoint-{global_step}.pth")
                torch.save({
                    "global_step": global_step,
                    "phase": phase,
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "gaze_mlp_state_dict": accelerator.unwrap_model(gaze_mlp).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                }, ckpt_path)
                logger.info(f"Saved checkpoint: {ckpt_path}")

                ckpts = sorted(
                    [f for f in os.listdir(save_dir)
                     if f.startswith("checkpoint-") and f.endswith(".pth")],
                    key=lambda x: int(x.split("-")[1].split(".")[0]),
                )
                while len(ckpts) > cfg.total_limit:
                    os.remove(os.path.join(save_dir, ckpts.pop(0)))

    if accelerator.is_main_process:
        final = os.path.join(save_dir, "final_model.pth")
        torch.save({
            "phase": phase,
            "model_state_dict": accelerator.unwrap_model(model).state_dict(),
            "gaze_mlp_state_dict": accelerator.unwrap_model(gaze_mlp).state_dict(),
        }, final)
        logger.info(f"Training complete — {final}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    main(cfg, args.config)
