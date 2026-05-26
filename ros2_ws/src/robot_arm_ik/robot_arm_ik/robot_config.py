PANDA_JOINTS = [
    'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
    'panda_joint5', 'panda_joint6', 'panda_joint7',
]
BASE_FOOTPRINT_TO_BASE_LINK_Z = 0.0


def get_a2_config():
    return {
        'left_arm':  {'joint_names': PANDA_JOINTS, 'fk_chain': None},
        'right_arm': {'joint_names': [],            'fk_chain': None},
    }
