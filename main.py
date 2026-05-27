import os
os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)
os.environ["HF_HUB_OFFLINE"] = "1"

CACHE_DIR = "/mimer/NOBACKUP/groups/zijian/common/sam3_model"
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", CACHE_DIR + "/hub")
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR + "/transformers")
os.environ.setdefault("TORCH_HOME", CACHE_DIR)

import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
 
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor
from lavila.datasets import VideoCaptionDatasetBase 



DEFAULTS = dict(
    ek_video_root="/mimer/NOBACKUP/groups/zijian/common/datasets/EK100_256p",        # dir globbed as <root>/*/*.MP4
    ek_metadata=("/mimer/NOBACKUP/groups/zijian/common/lavila_workspace/datasets/EK100/epic-kitchens-100-annotations/"
                 "EPIC_100_validation.csv"),
    dataset="ek100_cls",                                 # LaViLa dataset key
    clip_length=16,                                      # frames sampled per segment
    clip_stride=2,                                       # unused by ek100_cls, kept for API
    num_workers=4,
 
    # --- SAM3 ---
    model_id="facebook/sam3",
    prompts=["hands"],                                   # prompts will be replaced by smartclip outputs
    image_size=1008,                                     # SAM3 default; lower = faster/worse
 
    # --- feature extraction knobs (see header) ---
    vision_encoder_module=None,                          # None => auto-discover  (VERIFY)
    feature_channels_last=True,                          # (B,H,W,C) => True       (VERIFY)
    save_dense=False,                                    # also dump per-frame dense maps (big!)
 
    # --- output ---
    out_dir="sam3_features_ek100",
    max_clips=None,                                      # cap for a quick smoke test
)
 
device = Accelerator().device
dtype = torch.bfloat16

def build_lavila_dataset(cfg):
    """Wrap LaViLa's VideoCaptionDatasetBase so __getitem__ returns raw frames
    plus full metadata (narration/verb/noun) instead of just a class label."""
 
    class EK100ClipDataset(VideoCaptionDatasetBase):
        def __init__(self, dataset, root, metadata, clip_length, clip_stride):
            # is_trimmed=True => one sample per trimmed action segment
            super().__init__(dataset, root, metadata, is_trimmed=True)
            self.clip_length = clip_length
            self.clip_stride = clip_stride
 
        def __getitem__(self, i):
            # get_raw_item handles decord decoding + uniform frame sampling.
            # is_training=False => deterministic (centered) frame sampling.
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
            # frames: (T, H, W, 3) float32 in [0,255]
            return frames, meta
 
    ds = EK100ClipDataset(cfg.dataset, cfg.ek_video_root, cfg.ek_metadata,
                          cfg.clip_length, cfg.clip_stride)
    print(f"[data] LaViLa dataset '{cfg.dataset}': {len(ds)} trimmed segments")
    return ds



if __name__ == "__main__":

    class Cfg:
        pass

    cfg = Cfg()
    for k, v in DEFAULTS.items():
        setattr(cfg, k, v)

    ds = build_lavila_dataset(cfg)

    print("\n========== DATASET INFO ==========")
    print("Dataset length:", len(ds))

    # check first sample
    frames, meta = ds[0]

    print("\n========== FIRST SAMPLE ==========")

    print("\n--- Frames ---")
    print("Type:", type(frames))
    print("Shape:", frames.shape)      # expected: (T, H, W, 3)
    print("Dtype:", frames.dtype)
    print("Min:", frames.min().item())
    print("Max:", frames.max().item())

    print("\n--- Metadata ---")
    for k, v in meta.items():
        print(f"{k}: {v}")

    # inspect several samples
    # print("\n========== FEW EXAMPLES ==========")

    # for idx in [0, 1, 2]:
    #     frames, meta = ds[idx]

    #     print(f"\nSample {idx}")
    #     print("frames shape:", frames.shape)
    #     print("narration:", meta["narration"])
    #     print("verb:", meta["verb"])
    #     print("noun:", meta["noun"])

    # load sam3 video model and processor
    model_id = "facebook/sam3"
    model = Sam3VideoModel.from_pretrained(model_id, cache_dir=CACHE_DIR, local_files_only=True).to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained(model_id, cache_dir=CACHE_DIR, local_files_only=True)
    print("DEVICE: ", device)

    # Sam3 module names
    print("\n========== SAM3 MODULES ==========")
    for name, module in model.named_modules():
        print(name)

    # conver frames to uint8 
    video_frames = []
    for frame in frames:
        # frame shape: (H,W,3)
        frame = frame.cpu().numpy().astype(np.uint8)
        video_frames.append(frame)

    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.bfloat16,
        )
    text = "plate"
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=text,
    )

    # run frames in sam3 
    outputs_per_frame = {}
    print("\n========== SAM3 OUTPUTS ==========")
    for model_outputs in model.propagate_in_video_iterator(
        inference_session=inference_session
    ):
        processed_outputs = processor.postprocess_outputs(
            inference_session,
            model_outputs
        )
        idx = model_outputs.frame_idx
        outputs_per_frame[idx] = processed_outputs
        print(f"\nFrame {idx}")
        print("keys:", processed_outputs.keys())
        for k, v in processed_outputs.items():
            if torch.is_tensor(v):
                print(f"{k}: tensor shape={v.shape} dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"{k}: list length={len(v)}")
            else:
                print(f"{k}: type={type(v)}")
    print("\nProcessed frames:", len(outputs_per_frame))
 

