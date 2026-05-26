import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest


_QOS_LATCHED = QoSProfile(
    depth=2,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class SimArmIKNode(Node):
    """
    MoveIt2-backed simulation replacement for the proprietary arm_ik_node.

    Subscribes to arm_ik_node/cartesian_target (PoseStamped), calls MoveIt2's
    /compute_ik service, and re-publishes the solution on arm_ik_node/last_solution
    (JointState, TRANSIENT_LOCAL) so ik_ws.py can broadcast it to the browser.

    Node name is arm_ik_node so the webserver's arm_ik_node/set_parameters service
    reference is satisfied automatically.
    """

    def __init__(self):
        super().__init__('arm_ik_node')

        self.declare_parameter('il_pipeline.robot.joint_names',
                               ['panda_joint1', 'panda_joint2', 'panda_joint3',
                                'panda_joint4', 'panda_joint5', 'panda_joint6',
                                'panda_joint7'])
        self.declare_parameter('il_pipeline.robot.eef_link', 'panda_link8')
        self.declare_parameter('il_pipeline.robot.base_frame', 'world')

        self._joint_names = self.get_parameter(
            'il_pipeline.robot.joint_names').get_parameter_value().string_array_value
        self._eef_link = self.get_parameter(
            'il_pipeline.robot.eef_link').get_parameter_value().string_value
        self._base_frame = self.get_parameter(
            'il_pipeline.robot.base_frame').get_parameter_value().string_value

        self._ik_client = self.create_client(GetPositionIK, '/compute_ik')

        self._target_pub = self.create_publisher(PoseStamped, 'last_target', _QOS_LATCHED)
        self._solution_pub = self.create_publisher(JointState, 'last_solution', _QOS_LATCHED)

        self._target_sub = self.create_subscription(
            PoseStamped, 'cartesian_target', self._on_target, 10)

        self.get_logger().info(
            f'sim_arm_ik_node ready  eef={self._eef_link}  '
            f'joints={len(self._joint_names)}'
        )

    def _on_target(self, msg: PoseStamped):
        # Echo the target immediately with left_arm frame_id so ik_ws.py resolves arm='left'
        echo = PoseStamped()
        echo.header.stamp = self.get_clock().now().to_msg()
        echo.header.frame_id = 'left_arm'
        echo.pose = msg.pose
        self._target_pub.publish(echo)

        if not self._ik_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('compute_ik service not available — skipping IK request')
            return

        req = GetPositionIK.Request()
        ik_req = PositionIKRequest()
        ik_req.group_name = 'panda_arm'
        ik_req.robot_state.joint_state.name = list(self._joint_names)
        ik_req.robot_state.joint_state.position = [0.0] * len(self._joint_names)
        ik_req.avoid_collisions = True
        ik_req.pose_stamped.header.frame_id = self._base_frame
        ik_req.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        ik_req.pose_stamped.pose = msg.pose
        ik_req.timeout.sec = 1
        req.ik_request = ik_req

        future = self._ik_client.call_async(req)
        future.add_done_callback(self._ik_done)

    def _ik_done(self, future):
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().warn(f'IK call failed: {exc}')
            return

        # error_code 1 == SUCCESS
        if resp.error_code.val != 1:
            self.get_logger().warn(f'IK returned error_code {resp.error_code.val} — no solution published')
            return

        js = resp.solution.joint_state
        # Reorder positions to match our configured joint_names order
        pos_map = dict(zip(js.name, js.position))
        positions = [pos_map.get(n, 0.0) for n in self._joint_names]

        solution = JointState()
        solution.header.stamp = self.get_clock().now().to_msg()
        solution.header.frame_id = 'left_arm'
        solution.name = list(self._joint_names)
        solution.position = positions
        self._solution_pub.publish(solution)


def main(args=None):
    rclpy.init(args=args)
    node = SimArmIKNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
