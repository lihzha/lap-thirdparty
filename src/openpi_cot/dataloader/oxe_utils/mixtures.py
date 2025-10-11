OXE_NAMED_MIXTURES: dict[str, list[tuple[str, float]]] = {
    # === Bridge V2 Dataset ===
    "bridge": [
        ("bridge_v2_oxe", 1.0),  # Version of Bridge V2 in Open-X GCP Bucket
        # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],
    "droid": [
        ("droid", 1.0),
    ],
    # === UniVLA Magic Soup+ ===
    "omni_magic_soup_plus": [
        ("fractal20220817_data", 0.5),
        ("kuka", 0.1),
        ("bridge_v2_oxe", 1.0),
        ("taco_play", 2.0),
        ("jaco_play", 1.0),
        ("berkeley_cable_routing", 1.0),
        ("roboturk", 2.0),
        ("viola", 2.0),
        ("berkeley_autolab_ur5", 2.0),
        ("toto", 1.0),
        ("language_table", 0.1),
        ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
        ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),
        ("cmu_stretch", 1.0),
        ("bc_z", 0.2),
        ("fmb", 1.0),
        ("dobbe", 0.2),  # question
        ## Datasets for Navigation
        ("berkeley_gnm_recon", 1.0),
        ("berkeley_gnm_cory_hall", 1.0),
        ("berkeley_gnm_sac_son", 1.0),
    ],
    # "oxe_pi_magic_soup": [
    #     # ("droid", 2.0),
    #     # ("aloha", 0.4),
    #     # ("bridge_v2_oxe", 0.1),
    #     ("fmb", 0.071),
    #     # ("kuka", 0.05),
    #     # ("taco_play", 2.000),
    #     # ("furniture_bench_dataset_converted_externally_to_rlds", 0.024),
    #     # ("toto", 0.020),
    #     # ("austin_sirius_dataset_converted_externally_to_rlds", 0.017),
    #     # ("berkeley_autolab_ur5", 0.012),
    #     # ("viola", 0.009),
    #     # ("nyu_franka_play_dataset_converted_externally_to_rlds", 0.008),
    #     # ("berkeley_fanuc_manipulation", 0.007),
    #     # ("jaco_play", 0.004),
    #     # ("berkeley_cable_routing", 0.002),
    #     # ("cmu_stretch", 0.002),
    #     # # New
    #     # ("fractal20220817_data", 0.05),
    #     ("bc_z", 0.05),
    #     ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    #     ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
    #     ("roboturk", 2.0),
    # ],
    "oxe_pi_magic_soup": [
        ("bc_z", 0.5),
        ("droid", 0.5),
        ("fractal20220817_data", 1.0),
        # # ("kuka", 0.1),  # no language instructions, 580392 trajs
        ("bridge_v2_oxe", 1.0),
        ("taco_play", 1.0),
        ("jaco_play", 1.0),
        # ("berkeley_cable_routing", 1.0),  # no language instructions, only joint pos, 1482 trajs
        ("roboturk", 2.0),  # no prio, 2144 trajs
        # # ("viola", 2.0),  # gripper mostly out of view, 135 trajs
        ("berkeley_autolab_ur5", 1.0),
        # # ("toto", 1.0),   # no language instructions, 901 trajs
        # # ("language_table", 0.1),  442226 trajs
        # ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0), 550 trajs
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 50 trajs
        # # ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),  # no language instructions, 456 trajs
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.5),
        # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # only joint state
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 250 trajs
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 600 trajs
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),  # has empty language instructions, euler is zxy
        # ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),  # only joint state, 520 trajs
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),  # not sure quaternion is xyzw or wxyz
        ("cmu_stretch", 2.0),  # almost all movements are "move up"
        ("fmb", 1.0),
        ("dobbe", 0.2),  # question
        # ("sample_r1_lite", 0.2),  # only joint position available for both actions and states
        # ("agibot_dataset", 0.2),
    ],
    "oxe_pi_magic_soup_with_other_states_with_bimanual": [
        ("bc_z", 0.5),
        ("droid", 0.5),
        ("fractal20220817_data", 1.0),
        ("bridge_v2_oxe", 1.0),
        ("taco_play", 1.0),
        ("jaco_play", 1.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.5),
        ("utaustin_mutex", 1.0),
        ("berkeley_fanuc_manipulation", 2.0),  # not sure quaternion is xyzw or wxyz
        ("cmu_stretch", 2.0),  # almost all movements are "move up"
        ("fmb", 1.0),
        ("dobbe", 0.2),  # question
        ("berkeley_autolab_ur5", 1.0),
        ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),  # has empty language instructions, euler is zxy
        ("roboturk", 2.0),  # no prio, 2144 trajs. delta actions slightly sketchy sometimes.
        ### To be tested
        ("austin_buds_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 50 trajs
        ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 250 trajs
        ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),  # no language instructions, 600 trajs
        ### Bimanual
        # ("agibot_dataset", 0.2),
        ### TBD
        # ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),  # only joint state, 520 trajs. action scale is 300.
        # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # only joint state. action scale is 100000.
        # ("berkeley_cable_routing", 1.0),  # only joint pos, 1482 trajs, action scale incorrect and gripper not in view
        # # ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0), 550 trajs
        # # ("sample_r1_lite", 0.2),  # only joint position available for both actions and states
        # # # ("kuka", 0.1),  # no language instructions, 580392 trajs
        # # # ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),  # no language instructions, 456 trajs, no wrist camera
        # # # ("toto", 1.0),   # no language instructions, 901 trajs, no wrist camera
        # # # ("viola", 2.0),  # gripper mostly out of view, 135 trajs
        # # # ("language_table", 0.1),  442226 trajs
    ],
}


## TODO: to use ut_austin_mutex, we need to flip_wrist_image_channels,flip_image_channels. Other datasets are fine.
