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
        # ("fractal20220817_data", 0.5),
        # # ("kuka", 0.1),
        # ("bridge_v2_oxe", 1.0),
        # ("taco_play", 2.0),
        # ("jaco_play", 1.0),
        # # ("berkeley_cable_routing", 1.0),
        # # ("roboturk", 2.0),
        # ("viola", 2.0),
        # ("berkeley_autolab_ur5", 2.0),
        # # ("toto", 1.0),
        # # ("language_table", 0.1),
        # # ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
        # # ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
        # # ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
        ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
        # ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),  # joint state
        # # ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
        # # ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
        # ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),  # has empty language instructions
        # # ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),   # joint state
        # # ("utaustin_mutex", 1.0),  # joint state
        # ("berkeley_fanuc_manipulation", 2.0),   # join state
        # ("cmu_stretch", 1.0),  # almost all movements are "move up"
        # ("bc_z", 0.2),
        # ("fmb", 1.0),
        # # ("dobbe", 0.2),  # question
    ],
}


## TODO: to use ut_austin_mutex, we need to flip_wrist_image_channels,flip_image_channels. Other datasets are fine.
