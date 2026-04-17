#!/usr/bin/env python3
"""Standalone test-eval entrypoint for trained E2EGMNER checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e2egmner.data import Qwen25VLSFTCollator, TwitterGroundedMNERJsonl
from e2egmner.engine import run_test_eval
from e2egmner.utils.runtime import append_text_log, seed_everything, supports_bf16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint path (LoRA adapter or full model).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base model name/path used when checkpoint is LoRA adapter.")
    parser.add_argument("--test_jsonl", type=str, required=True, help="Path to test jsonl.")
    parser.add_argument("--image_root", type=str, default=None, help="Optional image root for relative image paths.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save predictions and metrics.")
    parser.add_argument("--tag", type=str, default="manual_eval", help="Tag suffix for prediction output file.")

    parser.add_argument("--device", type=str, default="cuda:0", help='e.g. "cuda", "cuda:0", "cpu".')
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_impl", type=str, default=None, choices=[None, "sdpa", "flash_attention_2"])

    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_iou", type=float, default=0.5)
    parser.add_argument("--no_strict_gold", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min_pixels", type=int, default=None, help="Override processor min_pixels.")
    parser.add_argument("--max_pixels", type=int, default=None, help="Override processor max_pixels.")
    parser.add_argument("--system_prompt", type=str, default=None)
    return parser.parse_args()


def resolve_precision(args: argparse.Namespace, is_cuda_device: bool) -> tuple[torch.dtype, bool, torch.dtype]:
    if args.bf16 and args.fp16:
        raise ValueError("Do not set both --bf16 and --fp16.")

    if not args.bf16 and not args.fp16:
        if is_cuda_device:
            args.bf16 = supports_bf16()
            args.fp16 = not args.bf16
        else:
            args.bf16 = False
            args.fp16 = False

    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    use_amp = is_cuda_device
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    return dtype, use_amp, autocast_dtype


def load_processor(args: argparse.Namespace, checkpoint_path: str):
    proc_kwargs = {"trust_remote_code": True}
    if args.min_pixels is not None:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        proc_kwargs["max_pixels"] = args.max_pixels

    ckpt_preproc = os.path.join(checkpoint_path, "preprocessor_config.json")
    source = checkpoint_path if os.path.exists(ckpt_preproc) else args.model_name
    return AutoProcessor.from_pretrained(source, **proc_kwargs)


def load_model(args: argparse.Namespace, checkpoint_path: str, dtype: torch.dtype):
    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    if args.attn_impl is not None:
        model_kwargs["attn_implementation"] = args.attn_impl

    adapter_cfg = os.path.join(checkpoint_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "Detected LoRA adapter checkpoint, but `peft` is not available. "
                "Please install peft first."
            ) from e
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
        model = PeftModel.from_pretrained(base, checkpoint_path, is_trainable=False)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(checkpoint_path, **model_kwargs)
    return model


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, verbose=False)

    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    output_dir = os.path.abspath(args.output_dir or os.path.join(checkpoint_path, "eval"))
    os.makedirs(output_dir, exist_ok=True)

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda_device = str(device).startswith("cuda")
    dtype, use_amp, autocast_dtype = resolve_precision(args, is_cuda_device)

    processor = load_processor(args, checkpoint_path)
    model = load_model(args, checkpoint_path, dtype=dtype)
    model.to(device)
    model.eval()

    test_dataset = TwitterGroundedMNERJsonl(args.test_jsonl, image_root=args.image_root)
    eval_collator = Qwen25VLSFTCollator(
        processor=processor,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        factor=28,
        do_jitter=False,
        system_prompt=args.system_prompt,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device,
        collate_fn=eval_collator,
        persistent_workers=(args.num_workers > 0),
    )

    print(
        f"[eval] checkpoint={checkpoint_path}  test_samples={len(test_dataset)}  "
        f"test_batch_size={args.test_batch_size}  test_batches={len(test_loader)}"
    )

    eval_log_path = os.path.join(output_dir, "eval_results.txt")
    append_text_log(eval_log_path, f"========== Eval start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========")
    append_text_log(
        eval_log_path,
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] checkpoint={checkpoint_path} tag={args.tag}",
    )

    metrics = run_test_eval(
        model=model,
        processor=processor,
        test_loader=test_loader,
        output_dir=output_dir,
        tag=args.tag,
        use_amp=use_amp,
        autocast_dtype=autocast_dtype,
        eval_max_new_tokens=args.eval_max_new_tokens,
        iou_threshold=args.eval_iou,
        strict_gold=(not args.no_strict_gold),
        log_file=eval_log_path,
    )

    metrics_path = os.path.join(output_dir, f"test_metrics_{args.tag}.json")
    payload = {
        "checkpoint": checkpoint_path,
        "model_name": args.model_name,
        "test_jsonl": os.path.abspath(args.test_jsonl),
        "image_root": args.image_root,
        "device": str(device),
        "dtype": str(dtype),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    append_text_log(eval_log_path, f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] metrics_json={metrics_path}")
    append_text_log(eval_log_path, f"========== Eval end {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========")
    print(f"[eval] metrics json saved: {metrics_path}")


# #debug用 在正式训练前请注释掉该代码
# import debugpy
# try:
#         # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#         pass

if __name__ == "__main__":
    main()
