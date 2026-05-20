"""
train_act.py — ACT (Action Chunking Transformer) training entry point.

Usage:
    python train_act.py                               # uses configs/act_config.yaml
    python train_act.py --config configs/act_config.yaml
    python train_act.py --episodes_dir ../data/episodes --checkpoint_dir ../data/models
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from datasets.dataset_loader import make_dataloaders, Normalizer
from models.act_policy import ACTPolicy

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

def build_model(cfg: dict, device: torch.device) -> ACTPolicy:
    mcfg  = cfg['model']
    dcfg  = cfg['data']
    model = ACTPolicy(
        num_joints=mcfg.get('joint_input_dim', 7),
        latent_dim=mcfg.get('latent_dim', 32),
        chunk_size=dcfg.get('chunk_size', 10),
        d_model=mcfg.get('d_model', 256),
        nhead=mcfg.get('nhead', 8),
        num_encoder_layers=mcfg.get('num_encoder_layers', 4),
        num_decoder_layers=mcfg.get('num_decoder_layers', 7),
        dim_feedforward=mcfg.get('dim_feedforward', 2048),
        dropout=mcfg.get('dropout', 0.1),
        use_images=mcfg.get('use_images', False),
    )
    return model.to(device)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(
    model: ACTPolicy,
    loader,
    device: torch.device,
    kl_weight: float,
) -> tuple[float, float, float]:
    """Returns (avg_l1, avg_kl, avg_total)."""
    model.eval()
    total_l1 = total_kl = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            obs    = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            target = batch['action'].to(device)   # (B, chunk_size, num_joints)
            pred, kl = model(obs, action_chunk=target)
            total_l1 += F.l1_loss(pred, target).item()
            total_kl += kl.item()
            n += 1
    model.train()
    n = max(n, 1)
    avg_l1 = total_l1 / n
    avg_kl = total_kl / n
    return avg_l1, avg_kl, avg_l1 + kl_weight * avg_kl


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    epoch: int,
    model: ACTPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler,
    val_loss: float,
    config: dict,
    normalizer: Normalizer,
    num_joints: int,
    chunk_size: int,
) -> None:
    torch.save(
        {
            'epoch':       epoch,
            'algorithm':   'act',
            'num_joints':  num_joints,
            'chunk_size':  chunk_size,
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
    dcfg  = cfg['data']
    tcfg  = cfg['training']
    mcfg  = cfg['model']

    chunk_size  = dcfg.get('chunk_size', 10)
    kl_weight   = tcfg.get('kl_weight', 10.0)
    num_joints  = mcfg.get('joint_input_dim', 7)

    # ── Data ──
    print('Loading dataset...')
    train_loader, val_loader, normalizer = make_dataloaders(
        episodes_dir=dcfg['episodes_dir'],
        val_split=dcfg.get('val_split', 0.1),
        batch_size=tcfg['batch_size'],
        record_images=dcfg.get('record_images', False),
        chunk_size=chunk_size,
        action_key='joint_pos_abs',
        num_workers=0,
    )
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    print(f'  train steps: {n_train}  val steps: {n_val}  chunk_size: {chunk_size}')

    # ── Model ──
    model  = build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  ACTPolicy  trainable params: {n_params:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg['lr'],
        weight_decay=tcfg.get('weight_decay', 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=tcfg['epochs'],
        eta_min=tcfg['lr'] * 0.01,
    )
    grad_clip  = tcfg.get('grad_clip', 1.0)
    log_every  = tcfg.get('log_every_n_steps', 50)
    save_every = tcfg.get('save_every_n_epochs', 10)

    # ── Checkpointing ──
    ckpt_dir = Path(tcfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard ──
    writer = None
    if _TB:
        writer = SummaryWriter(log_dir=str(ckpt_dir / 'tb_logs'))

    best_val   = float('inf')
    global_step = 0
    t0 = time.time()

    print(f'Training ACT for {tcfg["epochs"]} epochs  kl_weight={kl_weight}...')

    for epoch in range(1, tcfg['epochs'] + 1):
        model.train()
        epoch_l1 = epoch_kl = 0.0
        n_batches = 0

        it = train_loader
        if _TQDM:
            it = tqdm(train_loader, desc=f'Epoch {epoch:3d}', leave=False, unit='batch')

        for batch in it:
            obs    = {k: v.to(device) for k, v in batch.items() if k != 'action'}
            target = batch['action'].to(device)   # (B, chunk_size, num_joints)

            optimizer.zero_grad()
            pred, kl = model(obs, action_chunk=target)
            l1   = F.l1_loss(pred, target)
            loss = l1 + kl_weight * kl
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_l1  += l1.item()
            epoch_kl  += kl.item()
            n_batches += 1
            global_step += 1

            if writer and global_step % log_every == 0:
                writer.add_scalar('train/l1_loss_step',    l1.item(),   global_step)
                writer.add_scalar('train/kl_loss_step',    kl.item(),   global_step)
                writer.add_scalar('train/total_loss_step', loss.item(), global_step)

            if _TQDM:
                it.set_postfix(l1=f'{l1.item():.4f}', kl=f'{kl.item():.4f}')

        scheduler.step()
        n = max(n_batches, 1)
        avg_l1  = epoch_l1 / n
        avg_kl  = epoch_kl / n
        val_l1, val_kl, val_total = validate(model, val_loader, device, kl_weight)
        lr_now = scheduler.get_last_lr()[0]

        print(
            f'  Epoch {epoch:3d}/{tcfg["epochs"]}  '
            f'l1={avg_l1:.4f}  kl={avg_kl:.4f}  '
            f'val_total={val_total:.4f}  '
            f'lr={lr_now:.2e}  {time.time()-t0:.0f}s'
        )

        if writer:
            writer.add_scalar('train/l1_loss',    avg_l1,     epoch)
            writer.add_scalar('train/kl_loss',    avg_kl,     epoch)
            writer.add_scalar('train/total_loss', avg_l1 + kl_weight * avg_kl, epoch)
            writer.add_scalar('val/l1_loss',      val_l1,     epoch)
            writer.add_scalar('val/kl_loss',      val_kl,     epoch)
            writer.add_scalar('val/total_loss',   val_total,  epoch)
            writer.add_scalar('train/lr',         lr_now,     epoch)

        if val_total < best_val:
            best_val = val_total
            save_checkpoint(
                ckpt_dir / 'best.pt',
                epoch, model, optimizer, scheduler,
                val_total, cfg, normalizer, num_joints, chunk_size,
            )

        if epoch % save_every == 0:
            save_checkpoint(
                ckpt_dir / f'epoch_{epoch:04d}.pt',
                epoch, model, optimizer, scheduler,
                val_total, cfg, normalizer, num_joints, chunk_size,
            )

    save_checkpoint(
        ckpt_dir / 'final.pt',
        tcfg['epochs'], model, optimizer, scheduler,
        best_val, cfg, normalizer, num_joints, chunk_size,
    )
    if writer:
        writer.close()

    print('\nTraining complete.')
    print(f'  best val loss : {best_val:.4f}')
    print(f'  checkpoints   : {ckpt_dir.resolve()}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train ACT policy')
    p.add_argument('--config', default='configs/act_config.yaml')
    p.add_argument('--episodes_dir', default=None,
                   help='Override episodes_dir from config')
    p.add_argument('--checkpoint_dir', default=None,
                   help='Override checkpoint_dir from config')
    p.add_argument('--device', default='auto',
                   help='cpu | cuda | auto')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

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
