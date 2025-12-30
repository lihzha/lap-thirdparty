OXE_NAMED_MIXTURES: dict[str, list[tuple[str, float]]] = {
    "oxe_magic_soup": [
        # ("kuka", 0.01),  # no language instructions, 580392 trajs, no wrist, action scale is strange
        ("bc_z", 1.0),
        ("droid", 2.0),
        ("fractal20220817_data", 1.0),
        ("bridge_v2_oxe", 1.0),
        ("taco_play", 1.0),
        (
            "jaco_play",
            1.0,
        ),  # gripper state and action still seems incorrect. Action sometimes should be 1 but is 0. State seems random. Ignore for now.
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.5),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),  # not sure quaternion is xyzw or wxyz
        # (
        #     "cmu_stretch",
        #     2.0,
        # ),  # almost all movements are "move up". gripper actions always zero, use gripper state as action.
        ("fmb", 0.5),
        # ("dobbe", 0.2),  # question
        ("berkeley_autolab_ur5", 1.0),
        # ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),  # has empty language instructions, euler is zxy
        # ("roboturk", 2.0),  # no prio, 2144 trajs. delta actions slightly sketchy sometimes. loss to high.
        ### To be tested
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 50 trajs
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 250 trajs
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 600 trajs
        ("viola", 1.0),  # gripper mostly out of view, 135 trajs
        ("molmoact_dataset", 1.0),
        ### Bimanual
        # ("agibot_large_dataset", 0.2),
        # ("sample_r1_lite", 1.0),
        ### TBD
        # ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),  # only joint state, 520 trajs. action scale is 300.
        # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # only joint state. action scale is 100000.
        # ("berkeley_cable_routing", 1.0),  # only joint pos, 1482 trajs, action scale incorrect and gripper not in view
        # # ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0), 550 trajs
        # # # ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),  # no language instructions, 456 trajs, no wrist camera
        # # # ("toto", 1.0),   # no language instructions, 901 trajs, no wrist camera
        # # # ("language_table", 0.1),  442226 trajs
    ],
    "libero_finetune": [
        ("libero_10_no_noops", 1.0),
        ("libero_spatial_no_noops", 1.0),
        ("libero_object_no_noops", 1.0),
        ("libero_goal_no_noops", 1.0),
    ],
    "oxe_magic_soup_vqa": [
        # ("kuka", 0.01),  # no language instructions, 580392 trajs, no wrist, action scale is strange
        ("bc_z", 1.0),
        ("droid", 2.0),
        ("fractal20220817_data", 1.0),
        ("bridge_v2_oxe", 1.0),
        ("taco_play", 1.0),
        (
            "jaco_play",
            1.0,
        ),  # gripper state and action still seems incorrect. Action sometimes should be 1 but is 0. State seems random. Ignore for now.
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.5),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),  # not sure quaternion is xyzw or wxyz
        # (
        #     "cmu_stretch",
        #     2.0,
        # ),  # almost all movements are "move up". gripper actions always zero, use gripper state as action.
        ("fmb", 0.5),
        # ("dobbe", 0.2),  # question
        ("berkeley_autolab_ur5", 1.0),
        # ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),  # has empty language instructions, euler is zxy
        # ("roboturk", 2.0),  # no prio, 2144 trajs. delta actions slightly sketchy sometimes. loss to high.
        ### To be tested
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 50 trajs
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 250 trajs
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 600 trajs
        ("viola", 1.0),  # gripper mostly out of view, 135 trajs
        ("molmoact_dataset", 1.0),
        ("coco_captions", 1.0),  # 10% COCO caption samples
        ("vqa", 0.1),  # 10% vqa samples
        ("lvis", 0.01),
        ("pixmo_cap", 1.0),  # 10% COCO caption samples
        ("pixmo_point", 0.1),  # 10% vqa samples
        ("paco_lvis", 1.0),
        ("paco_ego4d", 1.0),
    ],
    "franka_no_droid": [
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.5),
        ("utaustin_mutex", 1.0),
        ("fmb", 1.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 50 trajs
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 250 trajs
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 600 trajs
        ("viola", 1.0),  # gripper mostly out of view, 135 trajs
        ("molmoact_dataset", 1.0),
    ],
    # === Individual Datasets (for isolated visualization/testing) ===
    "kuka": [("kuka", 1.0)],
    "bc_z": [("bc_z", 1.0)],
    "fractal20220817_data": [("fractal20220817_data", 1.0)],
    "bridge_v2_oxe": [("bridge_v2_oxe", 1.0)],
    "taco_play": [("taco_play", 1.0)],
    "jaco_play": [("jaco_play", 1.0)],
    "furniture_bench_dataset_converted_externally_to_rlds": [
        ("furniture_bench_dataset_converted_externally_to_rlds", 1.0)
    ],
    "utaustin_mutex": [("utaustin_mutex", 1.0)],
    "berkeley_fanuc_manipulation": [("berkeley_fanuc_manipulation", 1.0)],
    "cmu_stretch": [("cmu_stretch", 1.0)],
    "fmb": [("fmb", 1.0)],
    "dobbe": [("dobbe", 1.0)],
    "berkeley_autolab_ur5": [("berkeley_autolab_ur5", 1.0)],
    "dlr_edan_shared_control_converted_externally_to_rlds": [
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0)
    ],
    "bridge": [
        ("bridge_v2_oxe", 1.0),  # Version of Bridge V2 in Open-X GCP Bucket
        # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],
    "droid": [
        ("droid", 1.0),
    ],
    "roboturk": [("roboturk", 1.0)],
    "austin_buds_dataset_converted_externally_to_rlds": [("austin_buds_dataset_converted_externally_to_rlds", 1.0)],
    "austin_sailor_dataset_converted_externally_to_rlds": [("austin_sailor_dataset_converted_externally_to_rlds", 1.0)],
    "austin_sirius_dataset_converted_externally_to_rlds": [("austin_sirius_dataset_converted_externally_to_rlds", 1.0)],
    "viola": [("viola", 1.0)],
    "agibot_large_dataset": [("agibot_large_dataset", 1.0)],
    "sample_r1_lite": [("sample_r1_lite", 1.0)],
    "molmoact_dataset": [("molmoact_dataset", 1.0)],
    "planning_dataset": [("planning_dataset", 1.0)],
    "franka_dataset": [("franka_dataset", 1.0)],
    "berkeley_gnm_recon": [("berkeley_gnm_recon", 1.0)],
    "berkeley_gnm_sac_son": [("berkeley_gnm_sac_son", 1.0)],
    "berkeley_gnm_cory_hall": [("berkeley_gnm_cory_hall", 1.0)],
    "libero_10_no_noops": [("libero_10_no_noops", 1.0)],
    "libero_goal_no_noops": [("libero_goal_no_noops", 1.0)],
    "libero_object_no_noops": [("libero_object_no_noops", 1.0)],
    "libero_spatial_no_noops": [("libero_spatial_no_noops", 1.0)],
    # === VQA Datasets ===
    "coco_captions": [("coco_captions", 1.0)],
    "vqa": [("vqa", 1.0)],
    "pixmo_cap": [("pixmo_cap", 1.0)],
    "pixmo_point": [("pixmo_point", 1.0)],
    "lvis": [("lvis", 1.0)],
    "paco_lvis": [("paco_lvis", 1.0)],
    "paco_ego4d": [("paco_ego4d", 1.0)],
    "droid_100": [("droid_100", 1.0)],
}
## to use ut_austin_mutex, we need to flip_wrist_image_channels,flip_image_channels. Other datasets are fine.
