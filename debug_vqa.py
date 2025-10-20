"""Debug script to trace VQA inference and tokenization."""
import numpy as np
import jax.numpy as jnp
from openpi_cot.training import config as _config
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer
from openpi_cot.transforms import TokenizePromptAndReasoning

# Test tokenization
prompt = "Describe the image. Answer:"
print(f"Original prompt: {repr(prompt)}")

# Create tokenizer
tokenizer = PaligemmaCoTTokenizer(
    max_token_len=600,
    prompt_format="vqa",
    tokenizer_type="gemma3",
)

# Test tokenize_cot
tokens, pad_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(prompt, language_actions=None, state=None)

print(f"\nTokens: {tokens}")
print(f"Tokens shape: {tokens.shape}")
print(f"Pad mask: {pad_mask}")
print(f"Tokens (non-padded): {tokens[~pad_mask]}")

# Decode tokens back
decoded_prompt = tokenizer.decode(tokens.astype(np.int32))
print(f"\nDecoded prompt: {repr(decoded_prompt)}")

# Test TokenizePromptAndReasoning transform
transform = TokenizePromptAndReasoning(tokenizer)
data = {
    "prompt": prompt,
    "state": np.array([0.1, 0.2, 0.3]),
}
result = transform(data)
print(f"\nTransform result keys: {result.keys()}")
print(f"Tokenized prompt: {result['tokenized_prompt']}")
print(f"Tokenized prompt (non-padded): {result['tokenized_prompt'][~result['tokenized_prompt_mask']]}")
