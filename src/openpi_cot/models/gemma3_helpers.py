import jax.numpy as jnp
from openpi.shared import array_typing as at


def replace_image_placeholders(
    self,
    token_embeddings: at.Float[at.Array, "b s emb"],
    tokenized_sequence: at.Int[at.Array, "b s"],
    image_embeddings: at.Float[at.Array, "b n_img*n_patches emb"],
) -> at.Float[at.Array, "b s emb"]:
    """Replace placeholder tokens (-2) with actual image embeddings.

    Args:
        token_embeddings: Embeddings from tokenized sequence (includes placeholder embeddings)
        tokenized_sequence: Token IDs (includes -2 for placeholders)
        image_embeddings: Actual image embeddings from SigLIP [b, n_img*n_patches, emb]

    Returns:
        Updated embeddings with placeholders replaced
    """
    # Find placeholder positions: where token_id == -2
    is_placeholder = tokenized_sequence == -2  # [b, s]

    # Count total placeholders per batch element (should be same across batch)
    num_placeholders = jnp.sum(is_placeholder, axis=1)  # [b]

    # Create indices for scattering image embeddings into placeholder positions
    # For each batch element, we need to map placeholder positions to image embedding indices
    b, s, emb = token_embeddings.shape
    _, n_img_patches, _ = image_embeddings.shape

    # Build a mapping: for each position in sequence, what image index does it correspond to?
    # Cumulative sum of is_placeholder gives us the image token index for each placeholder
    placeholder_indices = jnp.cumsum(is_placeholder, axis=1) - 1  # [b, s], -1 to make 0-indexed

    # Clamp to valid range [0, n_img_patches)
    placeholder_indices = jnp.clip(placeholder_indices, 0, n_img_patches - 1)

    # Gather image embeddings according to placeholder indices
    # For each position, if it's a placeholder, get the corresponding image embedding
    # Otherwise, keep the original token embedding
    batch_indices = jnp.arange(b)[:, None]  # [b, 1]
    selected_image_embs = image_embeddings[batch_indices, placeholder_indices]  # [b, s, emb]

    # Replace: where is_placeholder, use image embedding; otherwise use token embedding
    result = jnp.where(
        is_placeholder[..., None],  # [b, s, 1] for broadcasting
        selected_image_embs,
        token_embeddings,
    )

    return result


def replace_placeholder_masks(
    self,
    text_mask: at.Bool[at.Array, "b s"],
    text_ar_mask: at.Bool[at.Array, "b s"],
    tokenized_sequence: at.Int[at.Array, "b s"],
    img_mask: at.Bool[at.Array, "b n_img"],
    img_ar_mask: at.Bool[at.Array, "b n_img"],
) -> tuple[at.Bool[at.Array, "b s"], at.Bool[at.Array, "b s"]]:
    """Replace placeholder positions in text masks with actual image masks.

    Args:
        text_mask: Mask from tokenized sequence (includes placeholder positions)
        text_ar_mask: AR mask from tokenized sequence (includes placeholder positions)
        tokenized_sequence: Token IDs (includes -2 for placeholders)
        img_mask: Actual image mask indicating which image tokens are valid
        img_ar_mask: Actual image AR mask for autoregressive masking

    Returns:
        (updated_mask, updated_ar_mask) with placeholder positions replaced
    """
    # Find placeholder positions: where token_id == -2
    is_placeholder = tokenized_sequence == -2  # [b, s]

    b, s = text_mask.shape
    _, n_img_patches = img_mask.shape

    # Build a mapping: for each position in sequence, what image index does it correspond to?
    placeholder_indices = jnp.cumsum(is_placeholder, axis=1) - 1  # [b, s], -1 to make 0-indexed
    placeholder_indices = jnp.clip(placeholder_indices, 0, n_img_patches - 1)

    # Gather image masks according to placeholder indices
    batch_indices = jnp.arange(b)[:, None]  # [b, 1]
    selected_img_mask = img_mask[batch_indices, placeholder_indices]  # [b, s]
    selected_img_ar_mask = img_ar_mask[batch_indices, placeholder_indices]  # [b, s]

    # Replace: where is_placeholder, use image mask; otherwise use text mask
    updated_mask = jnp.where(is_placeholder, selected_img_mask, text_mask)
    updated_ar_mask = jnp.where(is_placeholder, selected_img_ar_mask, text_ar_mask)

    return updated_mask, updated_ar_mask
