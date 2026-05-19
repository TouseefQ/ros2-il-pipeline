"""
train_bc.py — Behavior Cloning training entry point.

Usage:
    python train_bc.py                              # uses configs/bc_config.yaml
    python train_bc.py --config configs/bc_config.yaml
    python train_bc.py --config configs/bc_config.yaml --episodes_dir /data/episodes
    python train_bc.py --config configs/bc_config.yaml --device cuda
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Allow running from the training/ directory or repo root
sys.path.insert(0, str(Path(__file__).parent))

from datasets.dataset_loader import make_dataloaders, Normalizer
from models.bc_policy import MLPPolicy, CNNMLPPolicy

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(cfg: dict, device: torch.device) -> nn.Module:
    mcfg = cfg['model']
    num_joints = mcfg.get('joint_input_dim', 7)
    hidden_dims = mcfg.get('hidden_dims', [256, 256, 256])
    dropout = mcfg.get('dropout', 0.1)
    model_type = mcfg.get('type', 'mlp')

    if model_type == 'mlp':
        model = MLPPolicy(
            num_joints=num_joints,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif model_type == 'cnn_mlp':
        num_cameras = 1 + int(cfg['data'].get('record_images', False))
        model = CNNMLPPolicy(
            num_joints=num_joints,
            hidden_dims=hidden_dims,
            dropout=dropout,
            num_cameras=num_cameras,
        )
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return model.to(device)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            obs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            target = batch['action'].to(device)
            pred = model(obs)
            total_loss += criterion(pred, target).item()
            n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    val_loss: float,
    config: dict,
    normalizer: Normalizer,
    num_joints: int,
) -> None:
    torch.save(
        {
            'epoch':       epoch,
            'algorithm':   'bc',
            'num_joints':  num_joints,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict() if scheduler else None,
            'val_loss':    val_loss,
            'config':      config,
            'normalizer':  normalizer.state_dict(),
        },
        path,
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict, device: torch.device) -> None:
    dcfg = cfg['data']
    tcfg = cfg['training']

    # ── Data ──
    print('Loading dataset...')
    train_loader, val_loader, normalizer = make_dataloaders(
        episodes_dir=dcfg['episodes_dir'],
        val_split=dcfg.get('val_split', 0.1),
        batch_size=tcfg['batch_size'],
        record_images=dcfg.get('record_images', False),
        chunk_size=1,
        action_key='joint_pos_abs',
        num_workers=0,
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    num_joints = cfg['model'].get('joint_input_dim', 7)
    print(f'  train steps: {n_train}  val steps: {n_val}  joints: {num_joints}')

    # ── Model ──
    model = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  model: {cfg["model"]["type"]}  trainable params: {n_params:,}')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tcfg['lr'],
        weight_decay=tcfg.get('weight_decay', 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tcfg['epochs'],
        eta_min=tcfg['lr'] * 0.01,
    )
    criterion = nn.MSELoss()
    grad_clip = tcfg.get('grad_clip', 1.0)
    log_every = tcfg.get('log_every_n_steps', 50)
    save_every = tcfg.get('save_every_n_epochs', 10)

    # ── Checkpointing ──
    ckpt_dir = Path(tcfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard ──
    writer = None
    if _TB:
        tb_dir = ckpt_dir / 'tb_logs'
        writer = SummaryWriter(log_dir=str(tb_dir))

    best_val = float('inf')
    global_step = 0

    print(f'Training for {tcfg["epochs"]} epochs...')
    t0 = time.time()

    for epoch in range(1, tcfg['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        it = train_loader
        if _TQDM:
            it = tqdm(train_loader, desc=f'Epoch {epoch:3d}', leave=False, unit='batch')

        for batch in it:
            obs = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            target = batch['action'].to(device)

            optimizer.zero_grad()
            pred = model(obs)
            loss = criterion(pred, target)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            n_batches += 1
            global_step += 1

            if writer and global_step % log_every == 0:
                writer.add_scalar('train/loss_step', batch_loss, global_step)

            if _TQDM:
                it.set_postfix(loss=f'{batch_loss:.4f}')

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)
        val_loss   = validate(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]

        elapsed = time.time() - t0
        print(
            f'  Epoch {epoch:3d}/{tcfg["epochs"]}  '
            f'train={avg_train:.4f}  val={val_loss:.4f}  '
            f'lr={lr_now:.2e}  {elapsed:.0f}s'
        )

        if writer:
            writer.add_scalar('train/loss_epoch', avg_train, epoch)
            writer.add_scalar('val/loss',         val_loss,  epoch)
            writer.add_scalar('train/lr',          lr_now,    epoch)

        # Save best
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                ckpt_dir / 'best.pt',
                epoch, model, optimizer, scheduler, val_loss,
                cfg, normalizer, num_joints,
            )

        # Periodic save
        if epoch % save_every == 0:
            save_checkpoint(
                ckpt_dir / f'epoch_{epoch:04d}.pt',
                epoch, model, optimizer, scheduler, val_loss,
                cfg, normalizer, num_joints,
            )

    # Save final
    save_checkpoint(
        ckpt_dir / 'final.pt',
        tcfg['epochs'], model, optimizer, scheduler, best_val,
        cfg, normalizer, num_joints,
    )
    if writer:
        writer.close()

    print(f'\nTraining complete.')
    print(f'  best val loss : {best_val:.4f}')
    print(f'  checkpoints   : {ckpt_dir.resolve()}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train BC policy')
    p.add_argument('--config', default='configs/bc_config.yaml')
    p.add_argument('--episodes_dir', default=None,
                   help='Override episodes_dir from config')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Override checkpoint_dir from config')
    p.add_argument('--device', default='auto',
                   help='cpu | cuda | auto (auto picks cuda if available)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.episodes_dir:
        cfg['data']['episodes_dir'] = args.episodes_dir
    if args.checkpoint_dir:
        cfg['training']['checkpoint_dir'] = args.checkpoint_dir

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    train(cfg, device)


if __name__ == '__main__':
    main()
