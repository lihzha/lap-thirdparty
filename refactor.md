## pi_cot.py
1. Learn from third_party/openpi/src/openpi/models/pi0_fast.py: rewrite our sample_reasoning_tokens to be same as pi0_fast's sample_actions, but should work effectively the same. Note that tokens for our model are already left padded. Optimize our sample_actions so that it can first sample tokens, then generate actions with diffusion, so it's a combination of pi0_fast's sample_actions and pi0's sample actions.

2. Clean up compute_loss function to make it modular and consist of two types of loss: cross entropy loss and diffusion loss on actions. Users should be able to easily only use one loss for training, or use both losses, through config.


## 