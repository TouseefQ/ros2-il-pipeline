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

import os
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


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
        # TODO Stage 4: scan output_dir, count episodes, sum steps
        stats = {'episodes': 0, 'total_steps': 0, 'output_dir': self._output_dir}
        msg = String()
        msg.data = json.dumps(stats)
        self._stats_pub.publish(msg)

    def _scan_episodes(self) -> list[str]:
        # TODO Stage 4: return sorted list of .hdf5 paths
        if not os.path.isdir(self._output_dir):
            return []
        return sorted(
            f for f in os.listdir(self._output_dir) if f.endswith('.hdf5')
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DatasetManagerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
