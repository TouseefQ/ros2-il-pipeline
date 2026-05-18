"""
DatasetManagerNode — episode inventory management.

Services offered:
  ~/list_episodes    (ListEpisodes — TODO Stage 4)
  ~/delete_episode   (DeleteEpisode — TODO Stage 4)
  ~/export_dataset   (ExportDataset — TODO Stage 4)

Publishes:
  /il/dataset_stats   std_msgs/String  (JSON summary of episode count + total steps)

The node watches the output directory and keeps an in-memory index of
all .hdf5 episode files for fast querying.
"""

import json
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import h5py
    _H5PY_AVAILABLE = True
except ImportError:
    _H5PY_AVAILABLE = False


class DatasetManagerNode(Node):
    def __init__(self) -> None:
        super().__init__('dataset_manager')

        self.declare_parameter('il_pipeline.data_collection.output_dir', '/data/episodes')

        self._output_dir: str = self.get_parameter(
            'il_pipeline.data_collection.output_dir').value

        # ── Publisher ────────────────────────────────────────────────────────
        self._stats_pub = self.create_publisher(String, '/il/dataset_stats', 10)

        # ── Periodic stats publish ────────────────────────────────────────────
        self._timer = self.create_timer(10.0, self._publish_stats)

        self.get_logger().info(
            'DatasetManagerNode watching: %s' % self._output_dir)

    def _publish_stats(self) -> None:
        episodes = self._scan_episodes()

        total_steps = 0
        modes: dict[str, int] = {}
        for ep in episodes:
            total_steps += ep.get('num_steps', 0)
            mode = ep.get('collection_mode', 'unknown')
            modes[mode] = modes.get(mode, 0) + 1

        stats = {
            'episode_count': len(episodes),
            'total_steps': total_steps,
            'collection_modes': modes,
            'output_dir': self._output_dir,
        }
        msg = String()
        msg.data = json.dumps(stats)
        self._stats_pub.publish(msg)

    def _scan_episodes(self) -> list[dict]:
        if not os.path.isdir(self._output_dir):
            return []

        results: list[dict] = []
        paths = sorted(
            os.path.join(self._output_dir, f)
            for f in os.listdir(self._output_dir)
            if f.endswith('.hdf5')
        )

        if not _H5PY_AVAILABLE:
            # Fallback: count files without reading metadata
            return [{'episode_id': os.path.basename(p), 'num_steps': 0} for p in paths]

        for path in paths:
            try:
                with h5py.File(path, 'r') as f:
                    meta = f.get('meta', {})
                    results.append({
                        'episode_id': str(meta['episode_id'][()]) if 'episode_id' in meta else os.path.basename(path),
                        'num_steps': int(meta['num_steps'][()]) if 'num_steps' in meta else 0,
                        'collection_mode': str(meta['collection_mode'][()]) if 'collection_mode' in meta else 'unknown',
                        'path': path,
                    })
            except Exception as exc:
                self.get_logger().warn('Skipping %s: %s' % (path, str(exc)))

        return results


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
