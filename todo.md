Implement Paligemma2.
    1. Implement gemma2.py to support post_attention_norm, post_ffw_norm, final_logits_softcap and keep the other parts the same. Note you should always respect the MoE structure, and post_attention_norm, post_ffw_norm, attn_logits_softcap and final_logits_softcap will only apply to expert at position 0. 
    (Low priority) 2. Replace our sampling_reasoning_tokens in pi_cot.py with pi0_fast's sample_actions. To do this, you need to refer to gemma_fast.py and modify gemma2.py to use the _update_cache logic to avoid dynamic shapes in jax.while_loop, while still using the MoE structure. Keep in mind that in the MoE only the LLM part needs to do autoregressive decoding, and action expert do not need to do this.
    3. Implement the interfaces for gemma2 in pi_cot_config.py and gemma_adapter.py.

When use OXEDatasets calculate global normalization across all files.

Try using DROID-like wrist camera for our Franka robot

Evaluate VQA on Paligemma1. 