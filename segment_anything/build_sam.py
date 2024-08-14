import torch
from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )
def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )

sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}

def _resize_pos_embed(pos_embed, new_size, old_size, num_channels):
    # Reshape to match the old size
    pos_embed = pos_embed.reshape(1, old_size, old_size, num_channels).permute(0, 3, 1, 2)
    # Resize the positional embedding
    pos_embed = torch.nn.functional.interpolate(pos_embed, size=new_size, mode='bicubic', align_corners=False)
    # Reshape back to the original format
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_size[0], new_size[1], num_channels)
    return pos_embed


def _resize_rel_pos(rel_pos, new_size, old_size):
    max_rel_dist = 2 * max(new_size) - 1
    # Interpolate rel pos
    rel_pos_resized = torch.nn.functional.interpolate(
        rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
        size=max_rel_dist,
        mode="linear",
    )
    rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    return rel_pos_resized


def _build_sam(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 704  # Update the image size here
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        # Resize pos_embed
        old_size = state_dict['image_encoder.pos_embed'].shape[1]
        num_channels = state_dict['image_encoder.pos_embed'].shape[-1]
        new_size = (image_size // vit_patch_size, image_size // vit_patch_size)
        state_dict['image_encoder.pos_embed'] = _resize_pos_embed(state_dict['image_encoder.pos_embed'], new_size,
                                                                  old_size, num_channels)

        # Resize relative positional embeddings for each block
        for i in encoder_global_attn_indexes:
            state_dict[f'image_encoder.blocks.{i}.attn.rel_pos_h'] = _resize_rel_pos(
                state_dict[f'image_encoder.blocks.{i}.attn.rel_pos_h'], new_size, old_size
            )
            state_dict[f'image_encoder.blocks.{i}.attn.rel_pos_w'] = _resize_rel_pos(
                state_dict[f'image_encoder.blocks.{i}.attn.rel_pos_w'], new_size, old_size
            )

        sam.load_state_dict(state_dict)
    return sam


