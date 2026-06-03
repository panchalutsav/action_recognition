"""
For each clip we print:
    - LaViLa/EPIC metadata (vid_path, start/end frame, narration, verb, noun)
    - clip-level RAM++ tags (sigmoid scores averaged across the sampled frames)
    - optionally, per-frame top-3 tags (toggle SHOW_PER_FRAME)
"""

import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ram.models import ram_plus
from ram import get_transform
from lavila.datasets import VideoCaptionDatasetBase


DEFAULTS = dict(
    # --- LaViLa / EPIC-Kitchens ---
    ek_video_root="/mimer/NOBACKUP/groups/zijian/common/datasets/EK100_256p",
    ek_metadata=("/mimer/NOBACKUP/groups/zijian/common/lavila_workspace/datasets/"
                 "EK100/epic-kitchens-100-annotations/EPIC_100_validation.csv"),
    dataset="ek100_cls",
    clip_length=16,
    clip_stride=2,  # unused for ek100_cls, kept for API compatibility

    # --- RAM++ ---
    ram_pretrained="/mimer/NOBACKUP/groups/zijian/common/ram_plus_pretrained/ram_plus_swin_large_14m.pth",
    image_size=384,
    vit="swin_l",

    # --- test loop ---
    num_clips=5,            # how many EPIC segments to inspect
    top_k=10,               # how many tags to print per clip
    show_per_frame=False,   # also print per-frame top-3
)


def build_lavila_dataset(cfg):
    class EK100ClipDataset(VideoCaptionDatasetBase):
        def __init__(self, dataset, root, metadata, clip_length, clip_stride):
            super().__init__(dataset, root, metadata, is_trimmed=True)
            self.clip_length = clip_length
            self.clip_stride = clip_stride

        def __getitem__(self, i):
            frames, _ = self.get_raw_item(
                i, is_training=False,
                clip_length=self.clip_length,
                clip_stride=self.clip_stride,
            )
            vid_path, start_frame, end_frame, narration, verb, noun = self.samples[i]
            meta = dict(
                index=int(i), vid_path=vid_path,
                start_frame=int(start_frame), end_frame=int(end_frame),
                narration=narration, verb=int(verb), noun=int(noun),
            )
            # frames: (T, H, W, 3) float32 in [0, 255]
            return frames, meta

    ds = EK100ClipDataset(cfg.dataset, cfg.ek_video_root, cfg.ek_metadata,
                          cfg.clip_length, cfg.clip_stride)
    print(f"[data] LaViLa dataset '{cfg.dataset}': {len(ds)} trimmed segments")
    return ds


def frames_to_ram_batch(frames, transform, device):
    """LaViLa frames are (T, H, W, 3) float32 [0,255]; RAM's get_transform
    pipeline expects PIL images. Convert + transform + stack."""
    batch = []
    for f in frames:
        pil = Image.fromarray(f.cpu().numpy().astype(np.uint8))
        batch.append(transform(pil))
    return torch.stack(batch, dim=0).to(device)   # (T, 3, image_size, image_size)



@torch.no_grad()
def ram_scores_batched(image_tensor, model, device):
    image_embeds = model.image_proj(model.visual_encoder(image_tensor))
    image_atts   = torch.ones(image_embeds.size()[:-1],
                              dtype=torch.long, device=device)

    image_cls_embeds     = image_embeds[:, 0, :]
    image_spatial_embeds = image_embeds[:, 1:, :]
    bs = image_spatial_embeds.shape[0]

    des_per_class = int(model.label_embed.shape[0] / model.num_class)
    image_cls_embeds = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
    reweight_scale = model.reweight_scale.exp()

    logits_per_image = reweight_scale * image_cls_embeds @ model.label_embed.t()
    logits_per_image = logits_per_image.view(bs, -1, des_per_class)
    weight_normalized = F.softmax(logits_per_image, dim=2)

    label_embed_reweight = torch.empty(bs, model.num_class, 512,
                                       device=device, dtype=image_embeds.dtype)
    reshaped_value = model.label_embed.view(-1, des_per_class, 512)
    for i in range(bs):
        product = weight_normalized[i].unsqueeze(-1) * reshaped_value
        label_embed_reweight[i] = product.sum(dim=1)

    label_embed = F.relu(model.wordvec_proj(label_embed_reweight))

    # --- alignment decoder + per-class classifier ---
    tagging_embed = model.tagging_head(
        encoder_embeds=label_embed,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False, mode='tagging',
    )
    logits = model.fc(tagging_embed[0]).squeeze(-1)        # [B, num_class]

    scores = torch.sigmoid(logits).cpu().numpy()           # [B, num_class]
    if len(model.delete_tag_index) > 0:
        scores[:, model.delete_tag_index] = 0.0
    return scores


def print_top_k(scores_1d, tag_list, top_k, label):
    order = np.argsort(-scores_1d)
    print(f"  {label}:")
    for j in order[:top_k]:
        print(f"    {tag_list[j]:<24s} {scores_1d[j]:.4f}")



if __name__ == "__main__":
    class Cfg: pass
    cfg = Cfg()
    for k, v in DEFAULTS.items():
        setattr(cfg, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}")

    ds = build_lavila_dataset(cfg)

    print(f"[ram++] loading {cfg.ram_pretrained}")
    model = ram_plus(pretrained=cfg.ram_pretrained,
                     image_size=cfg.image_size,
                     vit=cfg.vit)
    model.eval().to(device)
    transform = get_transform(image_size=cfg.image_size)
    tag_list = list(model.tag_list)
    print(f"[ram++] num_class={model.num_class}  label_embed={tuple(model.label_embed.shape)}")

    n_clips = min(cfg.num_clips, len(ds))

    for idx in range(n_clips):
        frames, meta = ds[idx]
        print(f"\n========== clip {idx} ==========")
        for k, v in meta.items():
            print(f"  {k:<12s}: {v}")
        print(f"  frames shape: {tuple(frames.shape)}  dtype={frames.dtype}")

        batch = frames_to_ram_batch(frames, transform, device)   # [T, 3, H, W]
        scores = ram_scores_batched(batch, model, device)        # [T, num_class]

        # clip-level: mean over the T sampled frames
        clip_scores = scores.mean(axis=0)
        print_top_k(clip_scores, tag_list, cfg.top_k,
                    f"top-{cfg.top_k} tags  (mean over {scores.shape[0]} frames)")

        if cfg.show_per_frame:
            for t in range(scores.shape[0]):
                print_top_k(scores[t], tag_list, 3, f"frame {t}")