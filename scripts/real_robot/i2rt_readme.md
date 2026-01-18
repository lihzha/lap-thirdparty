In terminal 1:

source ~/i2rt/.venv/bin/activate
python ./scripts/real_robot/i2rt_main.py

Then press "n" for all images shown, except for the web camera one, we press "y"

Upper case the instruction, for example "Sort things into the basket"


In terminal 2, run the policy server:

Ours: uv run ./scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_boundsq99_32 --policy.dir checkpoints/pi_combined_cot_v6/ours_2048_rot_emastart10000_stateless_ki/15000 --policy.type=raw

Raw: uv run ./scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_boundsq99_32 --policy.dir checkpoints/pi_combined_cot_v6/oxetrue_eefchunk_newstats_noki/15000 --policy.type=raw

Ours + Co-train: uv run ./scripts/serve_policy.py policy:checkpoint --policy.config=paligemma_boundsq99 --policy.dir checkpoints/pi_combined_cot_v6/ours_2048_rot_cotrain_fixedactionloss_matchhorizon/15000 --policy.type=raw

