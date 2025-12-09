"""
configs.py

Defines per-dataset configuration (kwargs) for each dataset in Open-X Embodiment.

Configuration adopts the following structure:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB

    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth

    # Always 8-dim =>> changes based on `StateEncoding`
    state_obs_keys:
        StateEncoding.POS_EULER:    EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper Open/Close (1)
        StateEncoding.POS_QUAT:     EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
        StateEncoding.JOINT:        Joint Angles (7, <PAD> if fewer) + Gripper Open/Close (1)

    state_encoding: Type of `StateEncoding`
    action_encoding: Type of action encoding (e.g., EEF Position vs. Joint Position)
"""

from openpi_cot.datasets.utils.helpers import ActionEncoding
from openpi_cot.datasets.utils.helpers import StateEncoding

# === Individual Dataset Configs ===
OXE_DATASET_CONFIGS = {
    "fractal20220817_data": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        # "state_obs_keys": ["base_pose_tool_reached", "gripper_closed"],
        "state_obs_keys": ["eef_state", "gripper_closed"],
        # "state_encoding": StateEncoding.POS_QUAT,
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "kuka": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            # "clip_function_input/base_pose_tool_reached",
            # "gripper_closed",
            "state"
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    # "bridge_v2_oxe": {  # Version of Bridge V2 in Open X-Embodiment mixture
    #     "image_obs_keys": {"primary": "image", "secondary": "image_1", "wrist": None},
    #     "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    #     "state_obs_keys": ["EEF_state", None, "gripper_state"],
    #     "state_encoding": StateEncoding.POS_EULER,
    #     "action_encoding": ActionEncoding.EEF_POS,
    # },
    "bridge_v2_oxe": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ego4d_split_1": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ego4d_split_2": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ego4d_split_3": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ego4d_split_4": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bridge_orig": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bridge_dataset": {  # Original version of Bridge V2 from project website
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "taco_play": {
        "image_obs_keys": {
            "primary": "rgb_static",
            "secondary": None,
            "wrist": "rgb_gripper",
        },
        "depth_obs_keys": {
            "primary": "depth_static",
            "secondary": None,
            "wrist": "depth_gripper",
        },
        "state_obs_keys": ["state_eef", "state_gripper"],  # done
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "jaco_play": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state_eef", None, "state_gripper"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_cable_routing": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "top_image",
            "wrist": "wrist45_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["robot_state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "roboturk": {
        "image_obs_keys": {"primary": "front_rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_door_opening_surprising_effectiveness": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "viola": {
        "image_obs_keys": {
            "primary": "agentview_rgb",
            "secondary": None,
            "wrist": "eye_in_hand_rgb",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_autolab_ur5": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "hand_image",
        },
        "depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "toto": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "language_table": {
        "image_obs_keys": {"primary": "rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["effector_translation", None, None, None, None, None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "columbia_cairlab_pusht_real": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["robot_state", None, None, None, None, None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["ee_position", "ee_orientation", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image_additional_view",
            "wrist": None,
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": "depth_additional_view",
            "wrist": None,
        },
        "state_obs_keys": ["eef_state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": None,
            "wrist": "wrist_depth",
        },
        "state_obs_keys": ["tcp_pose", "gripper_state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "highres_image",
            "secondary": None,
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", None],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bc_z": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            # "present/xyz",
            # "present/axis_angle",
            "state",
            # "present/sensed_close",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image2",
            "wrist": "hand_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["end_effector_pose", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["pose_r", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "robo_net": {
        "image_obs_keys": {"primary": "image", "secondary": "image1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["pose", "gripper"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_pos", "gripper"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "asu_table_top_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "state_obs_keys": ["eef_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "imperialcollege_sawyer_wrist_cam": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, "state"],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["joint_state", "gripper_state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "uiuc_d3field": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "state_obs_keys": [None, None, None, None, None, None, None, None],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utaustin_mutex": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_fanuc_manipulation": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_playing_with_food": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "finger_vision_1",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_play_fusion": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_stretch": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["eef_state", "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },  # done
    "berkeley_gnm_recon": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_cory_hall": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_gnm_sac_son": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state", None, None],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "droid": {
        "image_obs_keys": {
            "primary": "exterior_image_1_left",
            "secondary": "exterior_image_2_left",
            "wrist": "wrist_image_left",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "cartesian_position",
            "gripper_position",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "fmb": {
        "image_obs_keys": {
            "primary": "image_side_1",
            "secondary": "image_side_2",
            "wrist": "image_wrist_2",
        },
        "depth_obs_keys": {
            "primary": "image_side_1_depth",
            "secondary": "image_side_2_depth",
            "wrist": "image_wrist_2_depth",
        },
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dobbe": {
        "image_obs_keys": {"primary": "wrist_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "roboset": {
        "image_obs_keys": {
            "primary": "image_left",
            "secondary": "image_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "rh20t": {
        "image_obs_keys": {
            "primary": "image_front",
            "secondary": "image_side_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": {  # "put carrot in bowl" task, 50 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_pour_corn_in_pot": {  # "pour corn from red bowl into steel pot" task, 50 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_flip_pot_upright": {  # "flip pot upright" task, 10 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_move_object_onto_plate": {  # "move <object> onto plate" task, 150 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_knock_object_over": {  # "knock <object> over" task, 70 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tdroid_cover_object_with_towel": {  # "cover <object> with towel" task, 45 demos @ 5 Hz control
        "image_obs_keys": {"primary": "static_image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "static_depth_image", "secondary": None, "wrist": None},
        "state_obs_keys": ["EEF_state", None, "gripper_state"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    ### DROID Finetuning datasets
    "droid_wipe": {
        "image_obs_keys": {"primary": "exterior_image_2_left", "secondary": None, "wrist": "wrist_image_left"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["proprio"],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_object_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_goal_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_10_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_10_no_noops_mini": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_goal_no_noops_mini": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_goal_no_noops_half": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_10_no_noops_half": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_goal_no_noops_quad": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_10_no_noops_quad": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "libero_combined": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],  # 8D: [EEF pose (6D), gripper (2D)]
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "sample_r1_lite": {
        "image_obs_keys": {
            "primary": "image_camera_head",
            "secondary": None,
            "wrist": "image_camera_wrist_left",
            "wrist_right": "image_camera_wrist_right",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            # "joint_position_torso",
            "joint_position_arm_left",
            "joint_position_arm_right",
            "gripper_state_left",
            "gripper_state_right",
        ],
        # "state_encoding": StateEncoding.JOINT_BIMANUAL,
        # "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "agibot_large_dataset": {
        "image_obs_keys": {
            "primary": "head_image",
            "secondary": None,
            "wrist": "image_camera_wrist_left",
            "wrist_right": "image_camera_wrist_right",
        },
        "depth_obs_keys": {
            "primary": None,
            "secondary": None,
            "wrist": "hand_left_image",
            "wrist_right": "hand_right_image",
        },
        "state_obs_keys": ["state"],
        # "state_encoding": StateEncoding.JOINT_BIMANUAL,
        # "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "molmoact_dataset": {
        "image_obs_keys": {
            "primary": "first_view_image",
            "secondary": "second_view_image",
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "planning_dataset": {
        "image_obs_keys": {
            "primary": "base_image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "franka_dataset": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": [
            "state",
        ],
        "state_encoding": StateEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
}


OXE_DATASET_METADATA = {
    "fractal20220817_data": {
        "control_frequency": 3,
        "language_annotations": "Templated",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "kuka": {
        "control_frequency": 10,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "bridge_v2_oxe": {
        "control_frequency": 5,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "taco_play": {
        "control_frequency": 15,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "jaco_play": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "berkeley_cable_routing": {
        "control_frequency": 10,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "roboturk": {
        "control_frequency": 5,  # 10
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "nyu_door_opening_surprising_effectiveness": {
        "control_frequency": 3,
        "language_annotations": "None",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "Yes",
    },
    "viola": {
        "control_frequency": 20,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "berkeley_autolab_ur5": {
        "control_frequency": 5,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "toto": {
        "control_frequency": 30,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "language_table": {
        "control_frequency": 10,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "columbia_cairlab_pusht_real": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
        "control_frequency": 3,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "control_frequency": 3,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "control_frequency": 2,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "control_frequency": 3,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "Templated",  # None
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "bc_z": {
        "control_frequency": 30,  # actually 10, but robot moves too slow
        "language_annotations": "Templated",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "Yes",
    },
    "usc_cloth_sim_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "utokyo_saytap_converted_externally_to_rlds": {
        "control_frequency": 50,
        "language_annotations": "Natural",
        "robot_morphology": "Quadrupedal Robot",
        "has_suboptimal": "No",
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Bi-Manual",
        "has_suboptimal": "No",
    },
    "robo_net": {
        "control_frequency": 1,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "control_frequency": 5,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "control_frequency": 30,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "control_frequency": "N/A, actions are run until robot comes to rest near the target position (quasistatic assumption)",
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "control_frequency": 5,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "asu_table_top_converted_externally_to_rlds": {
        "control_frequency": 12.5,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "control_frequency": 5,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "eth_agent_affordances": {
        "control_frequency": 66.6,
        "language_annotations": "Templated",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "Yes",
    },
    "imperialcollege_sawyer_wrist_cam": {
        "control_frequency": 10,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "control_frequency": 20,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "qut_dexterous_manipulation": {
        "control_frequency": 30,
        "language_annotations": "Natural",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "uiuc_d3field": {
        "control_frequency": 1,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "utaustin_mutex": {
        "control_frequency": 20,
        "language_annotations": "Natural Language annotations generate with GPT4 and followed by human correction.",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "berkeley_fanuc_manipulation": {
        "control_frequency": 10,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "cmu_playing_with_food": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "cmu_play_fusion": {
        "control_frequency": 5,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "cmu_stretch": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "berkeley_gnm_recon": {
        "control_frequency": 3,
        "language_annotations": "None",
        "robot_morphology": "Wheeled Robot",
        "has_suboptimal": "Yes",
    },
    "berkeley_gnm_cory_hall": {
        "control_frequency": 5,
        "language_annotations": "None",
        "robot_morphology": "Wheeled Robot",
        "has_suboptimal": "No",
    },
    "berkeley_gnm_sac_son": {
        "control_frequency": 10,
        "language_annotations": "None",
        "robot_morphology": "Wheeled Robot",
        "has_suboptimal": "No",
    },
    "robot_vqa": {
        "control_frequency": 10,
        "language_annotations": "Natural",
        "robot_morphology": "3 embodiments: single-armed robot, single-armed human, single-armed human using grasping tools",
        "has_suboptimal": "No",
    },
    "droid": {
        "control_frequency": 15,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "conq_hose_manipulation": {
        "control_frequency": 30,
        "language_annotations": "Natural",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "Yes",
    },
    "dobbe": {
        "control_frequency": 15,  # 3.75
        "language_annotations": "Natural",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "fmb": {
        "control_frequency": 10,
        "language_annotations": "Templated",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "io_ai_tech": {
        "control_frequency": 30,
        "language_annotations": "Templated",
        "robot_morphology": "Human",
        "has_suboptimal": "No",
    },
    "mimic_play": {
        "control_frequency": 15,
        "language_annotations": "Dataset does not contain language instruction annotation",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "aloha_mobile": {
        "control_frequency": 50,
        "language_annotations": "Templated",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "robo_set": {
        "control_frequency": 5,
        "language_annotations": "Natural",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "Yes",
    },
    "tidybot": {
        "control_frequency": "NaN",
        "language_annotations": "Our dataset specifies object placements in text form",
        "robot_morphology": "Mobile Manipulator",
        "has_suboptimal": "No",
    },
    "vima_converted_externally_to_rlds": {
        "control_frequency": "N/A due to use of primitive skills",
        "language_annotations": "Multimodal (image + language) templated instructions ",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "spoc": {
        "control_frequency": 10,
        "language_annotations": "Scripted language but augmented with LLMs",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "plex_robosuite": {
        "control_frequency": 20,
        "language_annotations": "None",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "sample_r1_lite": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "agibot_large_dataset": {
        "control_frequency": 30,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Bi-Manual",
        "has_suboptimal": "No",
    },
    "molmoact_dataset": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "planning_dataset": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "franka_dataset": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "libero_10_no_noops": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "libero_spatial_no_noops": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "libero_object_no_noops": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
    "libero_goal_no_noops": {
        "control_frequency": 15,
        "language_annotations": "Natual detailed instructions",
        "robot_morphology": "Single Arm",
        "has_suboptimal": "No",
    },
}

for dataset_name, metadata in OXE_DATASET_METADATA.items():
    if dataset_name in OXE_DATASET_CONFIGS:
        OXE_DATASET_CONFIGS[dataset_name].update(metadata)
