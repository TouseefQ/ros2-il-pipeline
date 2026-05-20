"""
dataset_viewer.py — visualize collected episodes from the HDF5 store.

Usage:
    python tools/dataset_viewer.py data/episodes/ep_000.hdf5
    python tools/dataset_viewer.py data/episodes/ep_000.hdf5 --show_images
    python tools/dataset_viewer.py data/episodes/ep_000.hdf5 --show_images --fps 5

Static view (default): joint trajectories, EEF XY path, EEF Z height,
gripper state, and action (joint_pos_abs) — all in one figure.

Image playback (--show_images): animated wrist/top camera frames with a
gripper state cursor. Requires record_images=true in the episode.
"""

import argparse
import sys
from pathlib import Path

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.animation import FuncAnimation
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


# ── Data loading ──────────────────────────────────────────────────────────────

def load_episode(path: Path) -> dict:
    with h5py.File(path, 'r') as f:
        meta = f['meta']

        def _s(key: str) -> str:
            v = meta[key][()]
            return v.decode() if isinstance(v, bytes) else str(v)

        obs = f['data/observation']
        act = f['data/action']

        data = {
            'episode_id': _s('episode_id') if 'episode_id' in meta else path.stem,
            'collection_mode': _s('collection_mode') if 'collection_mode' in meta else 'unknown',
            'num_steps': int(meta['num_steps'][()]),
            'record_images': bool(meta['record_images'][()]) if 'record_images' in meta else False,
            'timestamps': obs['timestamps'][()],
            'joint_pos': obs['joint_pos'][()],
            'joint_vel': obs['joint_vel'][()],
            'eef_pos': obs['eef_pos'][()],
            'eef_quat': obs['eef_quat'][()],
            'gripper_state': obs['gripper_state'][()].squeeze(),
            'joint_pos_abs': act['joint_pos_abs'][()],
            'images_wrist': None,
            'images_top': None,
        }

        if 'images' in obs:
            imgs = obs['images']
            if 'camera_wrist' in imgs:
                data['images_wrist'] = imgs['camera_wrist'][()]
            if 'camera_top' in imgs:
                data['images_top'] = imgs['camera_top'][()]

    return data


# ── Static plot ───────────────────────────────────────────────────────────────

def plot_static(data: dict) -> None:
    T = data['num_steps']
    t = data['timestamps'] - data['timestamps'][0]
    joint_pos = data['joint_pos']
    N = joint_pos.shape[1]
    eef = data['eef_pos']
    gripper = data['gripper_state']

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Episode: {data['episode_id']}  |  Mode: {data['collection_mode']}  |  {T} steps",
        fontsize=11, y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    # Joint positions
    ax_j = fig.add_subplot(gs[0, :])
    for i in range(N):
        ax_j.plot(t, joint_pos[:, i], label=f'j{i+1}', linewidth=0.9)
    ax_j.set_title('Joint Positions (rad)')
    ax_j.set_xlabel('Time (s)')
    ax_j.legend(loc='upper right', ncol=N, fontsize=7)
    ax_j.grid(True, alpha=0.3)

    # EEF XY top-down path
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_xy.plot(eef[:, 0], eef[:, 1], linewidth=0.9, color='steelblue')
    ax_xy.scatter(eef[0, 0], eef[0, 1], color='green', s=40, zorder=5, label='start')
    ax_xy.scatter(eef[-1, 0], eef[-1, 1], color='red', s=40, zorder=5, label='end')
    ax_xy.set_title('EEF Path — XY (top-down)')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.legend(fontsize=8)
    ax_xy.set_aspect('equal', adjustable='datalim')
    ax_xy.grid(True, alpha=0.3)

    # EEF Z height
    ax_z = fig.add_subplot(gs[1, 1])
    ax_z.plot(t, eef[:, 2], linewidth=0.9, color='darkorange')
    ax_z.set_title('EEF Height — Z')
    ax_z.set_xlabel('Time (s)')
    ax_z.set_ylabel('Z (m)')
    ax_z.grid(True, alpha=0.3)

    # Gripper state
    ax_g = fig.add_subplot(gs[2, 0])
    ax_g.plot(t, gripper, linewidth=0.9, color='purple')
    ax_g.set_title('Gripper State  (0 = closed, 1 = open)')
    ax_g.set_xlabel('Time (s)')
    ax_g.set_ylim(-0.05, 1.05)
    ax_g.grid(True, alpha=0.3)

    # Action vs observation (joint_pos_abs overlay)
    ax_a = fig.add_subplot(gs[2, 1])
    for i in range(N):
        ax_a.plot(t, data['joint_pos_abs'][:, i], linewidth=0.9, alpha=0.75)
    ax_a.set_title('Action — joint_pos_abs (rad)')
    ax_a.set_xlabel('Time (s)')
    ax_a.grid(True, alpha=0.3)

    plt.show()


# ── Animated image playback ───────────────────────────────────────────────────

def animate_images(data: dict, fps: float) -> None:
    wrist = data['images_wrist']
    top = data['images_top']

    if wrist is None and top is None:
        print('No image data in this episode — falling back to static view.')
        plot_static(data)
        return

    T = data['num_steps']
    t = data['timestamps'] - data['timestamps'][0]
    gripper = data['gripper_state']

    cam_panels = [(wrist, 'Wrist camera'), (top, 'Top camera')]
    cam_panels = [(arr, lbl) for arr, lbl in cam_panels if arr is not None]
    ncols = len(cam_panels) + 1  # camera panels + gripper state

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    fig.suptitle(f"Episode {data['episode_id']} — image playback  ({fps} fps)", fontsize=10)

    im_handles = []
    for col, (arr, lbl) in enumerate(cam_panels):
        im = axes[col].imshow(arr[0])
        axes[col].set_title(lbl)
        axes[col].axis('off')
        im_handles.append(im)

    # Gripper panel with running cursor
    ax_g = axes[-1]
    ax_g.plot(t, gripper, color='purple', linewidth=1.2, alpha=0.6)
    (cursor,) = ax_g.plot([], [], 'ro', markersize=6)
    ax_g.set_xlim(t[0], t[-1])
    ax_g.set_ylim(-0.05, 1.05)
    ax_g.set_title('Gripper state')
    ax_g.set_xlabel('Time (s)')
    ax_g.grid(True, alpha=0.3)

    def _update(frame: int):
        updates = []
        for im, (arr, _) in zip(im_handles, cam_panels):
            im.set_data(arr[frame])
            updates.append(im)
        cursor.set_data([t[frame]], [gripper[frame]])
        updates.append(cursor)
        return updates

    _anim = FuncAnimation(  # noqa: F841  (keep reference to prevent GC)
        fig, _update, frames=T, interval=1000.0 / fps, blit=True,
    )
    plt.tight_layout()
    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Visualize an episode HDF5 file')
    p.add_argument('episode', help='Path to .hdf5 episode file')
    p.add_argument('--show_images', action='store_true',
                   help='Animate camera frames (requires record_images=true)')
    p.add_argument('--fps', type=float, default=10.0,
                   help='Playback speed for --show_images (default: 10)')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not _H5PY_AVAILABLE:
        print('Error: h5py not installed — pip install h5py', file=sys.stderr)
        sys.exit(1)
    if not _MPL_AVAILABLE:
        print('Error: matplotlib/numpy not installed — pip install matplotlib numpy',
              file=sys.stderr)
        sys.exit(1)

    path = Path(args.episode)
    if not path.exists():
        print(f'File not found: {path}', file=sys.stderr)
        sys.exit(1)

    data = load_episode(path)

    print(f"Episode  : {data['episode_id']}")
    print(f"Mode     : {data['collection_mode']}")
    print(f"Steps    : {data['num_steps']}")
    print(f"Images   : {'yes' if data['record_images'] else 'no'}")

    if args.show_images:
        if not data['record_images']:
            print('Warning: episode was recorded without images — showing static view.')
            plot_static(data)
        else:
            animate_images(data, fps=args.fps)
    else:
        plot_static(data)


if __name__ == '__main__':
    main()
