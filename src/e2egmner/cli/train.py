"""Training entrypoint for Qwen2.5-VL LoRA fine-tuning on grounded multimodal NER."""

import argparse
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

try:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
except Exception:
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None

from ..data import Qwen25VLSFTCollator, TwitterGroundedMNERJsonl
from ..engine import run_dev_eval, run_test_eval
from ..utils.runtime import append_text_log, get_first_param_device, pick_vision_inputs, seed_everything, supports_bf16
from ..utils.tracking import _swan_finish, _swan_log, swanlab

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to the training jsonl file.")
    parser.add_argument(
        "--dev_jsonl",
        type=str,
        default=None,
        help="Optional dev jsonl used for validation after each epoch.",
    )

    parser.add_argument(
        "--test_jsonl",
        type=str,
        default=None,
        help="Optional test jsonl evaluated at the end of training.",
    )

    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen25vl_7b_lora")

    # device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='e.g. "cuda", "cuda:0", "cuda:1", "cpu". If None, auto choose.',
    )

    # batch / steps
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)

    # optimization
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_length", type=int, default=2048)

    # gradient clipping
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值（global norm）。<=0 表示关闭")

    # precision
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")

    # memory tricks
    parser.add_argument("--grad_ckpt", action="store_true")
    parser.add_argument("--attn_impl", type=str, default=None, choices=[None, "sdpa", "flash_attention_2"])

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=r"^(model\.language_model.*\.(v_proj|q_proj|k_proj|down_proj|o_proj|up_proj|gate_proj))$",
        help='Regex (recommended, aligns with ms-swift). If you pass "all-linear", it will be mapped to the default regex.',
    )

    # vision token control
    parser.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)

    # eval config
    parser.add_argument("--dev_batch_size", type=int, default=2)
    parser.add_argument("--test_batch_size", type=int, default=2)
    parser.add_argument("--eval_max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_iou", type=float, default=0.5)
    parser.add_argument("--no_strict_gold", action="store_true")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Optional system prompt used in both training and inference chat template. "
        "If None, a default prompt enforcing <think>/<answer> format is used.",
    )

    # ==========================
    # bbox jitter args (TRAIN ONLY)
    # ==========================
    parser.add_argument("--train_bbox_jitter", action="store_true", help="Enable bbox jitter for training collator only.")
    parser.add_argument("--jitter_beta", type=float, default=0.1, help="中心点扰动强度（相对 box 宽高比例）")
    parser.add_argument("--jitter_gamma", type=float, default=0.1, help="宽高缩放扰动强度（相对比例）")
    parser.add_argument("--jitter_iou_min", type=float, default=0.6, help="抖动后 bbox 与原 bbox 的最小 IoU 约束")
    parser.add_argument("--jitter_tries", type=int, default=5, help="IoU guard 采样重试次数")
    parser.add_argument("--jitter_min_area", type=int, default=32 * 32, help="小框面积阈值；小框会自动减弱扰动")
    parser.add_argument("--jitter_small_box_scale", type=float, default=0.5, help="小框扰动缩放系数")
    parser.add_argument("--jitter_dist", type=str, default="gauss", choices=["gauss", "uniform"], help="扰动分布；默认 gauss")
    parser.add_argument("--jitter_gauss_trunc_k", type=float, default=2.0, help="高斯扰动截断倍数 k")
    parser.add_argument("--jitter_scale_min", type=float, default=0.2, help="bbox 宽高缩放下界")
    parser.add_argument("--jitter_scale_max", type=float, default=5.0, help="bbox 宽高缩放上界")

    # ==========================
    # SwanLab args
    # ==========================
    parser.add_argument("--use_swanlab", action="store_true", help="Enable SwanLab experiment tracking")
    parser.add_argument("--swan_project", type=str, default="qwen25vl-gmner", help="SwanLab project name")
    parser.add_argument("--swan_workspace", type=str, default=None, help="SwanLab workspace (optional)")
    parser.add_argument("--swan_experiment_name", type=str, default=None, help="SwanLab experiment name (optional)")
    parser.add_argument("--swan_description", type=str, default=None, help="SwanLab description (optional)")
    parser.add_argument("--swan_tags", type=str, default=None, help='Comma-separated tags, e.g. "lora,gmner,debug"')
    parser.add_argument(
        "--swan_mode",
        type=str,
        default="cloud",
        choices=[None, "cloud", "local", "offline", "disabled"],
        help="SwanLab mode (optional)",
    )
    parser.add_argument("--swan_logdir", type=str, default=None, help="SwanLab logdir (optional)")
    parser.add_argument("--swan_id", type=str, default=None, help="Resume experiment id if needed")
    parser.add_argument(
        "--swan_resume",
        type=str,
        default=None,
        choices=[None, "allow", "must", "never"],
        help="Resume mode for SwanLab (optional)",
    )

    args = parser.parse_args()

    # 随机种子设置
    seed_everything(args.seed)

    # DataLoader worker 固定随机种子
    #让多进程加载可复现，且不同 worker 之间不会生成完全一样的随机序列
    def seed_worker(worker_id: int):
        worker_seed = args.seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    #固定主进程的随机种子 
    g = torch.Generator()
    g.manual_seed(args.seed) 

    # ==========================
    # SwanLab init (early)
    # ==========================
    swan_run = None
    if args.use_swanlab:
        if swanlab is None:
            raise RuntimeError("You set --use_swanlab but swanlab is not installed. Try: pip install swanlab")

        exp_name = args.swan_experiment_name
        if exp_name is None:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            exp_name = f"{os.path.basename(args.output_dir.rstrip('/'))}-{ts}"

        tags = None
        if args.swan_tags:
            tags = [t.strip() for t in args.swan_tags.split(",") if t.strip()]

        init_kwargs = dict(
            project=args.swan_project,
            workspace=args.swan_workspace,
            experiment_name=exp_name,
            description=args.swan_description,
            tags=tags,
            config=vars(args),
            mode=args.swan_mode,
            logdir=args.swan_logdir,
        )
        if args.swan_resume is not None:
            init_kwargs["resume"] = args.swan_resume
        if args.swan_id is not None:
            init_kwargs["id"] = args.swan_id

        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        swan_run = swanlab.init(**init_kwargs)
        print(
            f"[SwanLab] run started: project={args.swan_project}, "
            f"experiment_name={exp_name}, id={getattr(swan_run, 'id', None)}"
        )

    # -------- device / precision policy --------
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_cuda_device = str(device).startswith("cuda")

    if args.bf16 and args.fp16:
        raise ValueError("Do not set both --bf16 and --fp16. Choose one.")

    # 只根据“实际 device”决定是否启用 AMP / half precision，
    # 避免机器上虽然有 CUDA，但用户显式指定了 CPU 时仍错误启用 AMP。
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

    # -------- processor --------
    proc_kwargs = {"trust_remote_code": True}
    if args.min_pixels is not None:
        proc_kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        proc_kwargs["max_pixels"] = args.max_pixels
    processor = AutoProcessor.from_pretrained(args.model_name, **proc_kwargs) #多模态模型做输入预处理 会将多个模态的处理器组合起来

    # -------- model load --------
    model_kwargs = dict(
        dtype=dtype,
        trust_remote_code=True,
    )
    if args.attn_impl is not None:
        model_kwargs["attn_implementation"] = args.attn_impl

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)

    # 训练关掉 kv cache
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # -------- freeze base (LoRA-only) --------
    for p in model.parameters():
        p.requires_grad = False

    # -------- LoRA (align ms-swift) --------
    target = args.lora_target_modules
    if isinstance(target, str) and target.lower() in {"all-linear", "all_linear"}:
        target = r"^(model\.language_model.*\.(v_proj|q_proj|k_proj|down_proj|o_proj|up_proj|gate_proj))$"

    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target,
    )

    model = get_peft_model(model, lora_cfg)

    # -------- gradient checkpointing --------
    if args.grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()  #让输入经过 embedding 后得到的 hidden states 带上 requires_grad=True 把模型输入这一头重新接回计算图 否者反向传播会断

    # 确保 cache 关闭（LoRA wrap 后再确认一次）
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.train()
    print("Final LoRA target_modules:", model.peft_config["default"].target_modules)
    model.print_trainable_parameters()

    # data
    train_dataset = TwitterGroundedMNERJsonl(args.train_jsonl, image_root=args.image_root)

    train_collator = Qwen25VLSFTCollator(
        processor=processor,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        factor=28,
        system_prompt=args.system_prompt,
        do_jitter=bool(args.train_bbox_jitter),
        jitter_beta=float(args.jitter_beta),
        jitter_gamma=float(args.jitter_gamma),
        jitter_iou_min=float(args.jitter_iou_min),
        jitter_tries=int(args.jitter_tries),
        jitter_min_area=int(args.jitter_min_area),
        jitter_small_box_scale=float(args.jitter_small_box_scale),
        jitter_dist=str(args.jitter_dist),
        jitter_gauss_trunc_k=float(args.jitter_gauss_trunc_k),
        jitter_scale_min=float(args.jitter_scale_min),
        jitter_scale_max=float(args.jitter_scale_max),
    )

    eval_collator = Qwen25VLSFTCollator(
        processor=processor,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        factor=28,
        do_jitter=False,
        system_prompt=args.system_prompt,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=is_cuda_device,
        collate_fn=train_collator,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=(args.num_workers > 0),
    )

    dev_loader = None
    if args.dev_jsonl is not None:
        dev_dataset = TwitterGroundedMNERJsonl(args.dev_jsonl, image_root=args.image_root)
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=args.dev_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=is_cuda_device,
            collate_fn=eval_collator,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=(args.num_workers > 0),
        )
        print(f"[debug] dev_samples={len(dev_dataset)}  dev_batch_size={args.dev_batch_size}  dev_batches={len(dev_loader)}")

    test_loader = None
    if args.test_jsonl is not None:
        test_dataset = TwitterGroundedMNERJsonl(args.test_jsonl, image_root=args.image_root)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=is_cuda_device,
            collate_fn=eval_collator,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=(args.num_workers > 0),
        )
        print(f"[debug] test_samples={len(test_dataset)}  test_batch_size={args.test_batch_size}  test_batches={len(test_loader)}")

    # optimizer / scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]#可训练参数

    decay_params = [n for n, p in model.named_parameters() if "bias" not in n and "norm" not in n]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_params and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_params and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP（只在实际 CUDA device 上启用）
    use_amp = is_cuda_device
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and args.fp16))

    global_step = 0
    best_dev_score = -1.0
    best_epoch = -1
    best_state = None
    best_dir = os.path.join(args.output_dir, "best")

    # ✅ 统一文本日志文件
    eval_log_path = os.path.join(args.output_dir, "eval_results.txt")
    append_text_log(eval_log_path, f"========== Run start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========")
    append_text_log(
        eval_log_path,
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] system_prompt={repr(train_collator.system_prompt)}",
    )

    def optimizer_step_once(actual_accum_steps: int):
        """
        执行一次优化器更新。
        - 支持 epoch 尾部不足 grad_accum_steps 的剩余 micro-batches；
        - 尾部不足时会把梯度 rescale 到与常规累积步一致的量级。
        """
        nonlocal global_step

        if actual_accum_steps <= 0:
            return None

        grad_norm = None
        already_unscaled = False

        if actual_accum_steps != args.grad_accum_steps: #尾部 batch 不足，导致这次累积不满 grad_accum_steps的处理
            if scaler.is_enabled(): 
                scaler.unscale_(optimizer)  #手动修改梯度或者梯度裁剪之前把因为混合精度放大的梯度缩放回来
                already_unscaled = True

            rescale = args.grad_accum_steps / float(actual_accum_steps) #原先定好的accum_steps是剩余的步数的多少倍
            for p in trainable_params:
                if p.grad is not None:
                    p.grad.mul_(rescale) #放大这么多倍

        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            if scaler.is_enabled() and not already_unscaled:
                scaler.unscale_(optimizer)  
                already_unscaled = True
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        global_step += 1
        return grad_norm

    try:
        for epoch in range(args.epochs):
            pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
            optimizer.zero_grad(set_to_none=True)

            accum_counter = 0
            last_logged_loss = None

            for step, batch in enumerate(pbar):
                if batch is None:
                    continue

                target_device = get_first_param_device(model) #对齐设备
                for k, v in list(batch.items()):#把张量放到device上
                    if torch.is_tensor(v):
                        batch[k] = v.to(target_device, non_blocking=True)

                vision_inputs = pick_vision_inputs(batch)

                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    **vision_inputs,
                )
                loss = out.loss / args.grad_accum_steps

                if not torch.isfinite(loss):  #NaN 或 Inf跳过
                    print(f"[FATAL] non-finite loss at epoch={epoch} step={step} global_step={global_step}. loss={loss}")
                    return

                if scaler.is_enabled():#混合精度反穿
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum_counter += 1
                last_logged_loss = float(loss.item() * args.grad_accum_steps)

                if accum_counter == args.grad_accum_steps:
                    grad_norm = optimizer_step_once(actual_accum_steps=accum_counter)
                    accum_counter = 0

                    if swan_run is not None:
                        train_lr = float(scheduler.get_last_lr()[0])
                        log_data = {
                            "train/loss": last_logged_loss,
                            "train/lr": train_lr,
                        }
                        if grad_norm is not None:
                            log_data["train/grad_norm"] = float(grad_norm)
                        _swan_log(swan_run, log_data, step=global_step)

                    pbar.set_postfix(
                        loss=last_logged_loss,
                        lr=float(scheduler.get_last_lr()[0]),
                        grad_norm=(float(grad_norm) if grad_norm is not None else None),
                        step=global_step,
                    )

            # 处理 epoch 尾部不足 grad_accum_steps 的剩余 micro-batches
            if accum_counter > 0:
                grad_norm = optimizer_step_once(actual_accum_steps=accum_counter)

                if swan_run is not None:
                    train_lr = float(scheduler.get_last_lr()[0])
                    log_data = {
                        "train/loss": last_logged_loss if last_logged_loss is not None else 0.0,
                        "train/lr": train_lr,
                    }
                    if grad_norm is not None:
                        log_data["train/grad_norm"] = float(grad_norm)
                    _swan_log(swan_run, log_data, step=global_step)

                pbar.set_postfix(
                    loss=(last_logged_loss if last_logged_loss is not None else 0.0),
                    lr=float(scheduler.get_last_lr()[0]),
                    grad_norm=(float(grad_norm) if grad_norm is not None else None),
                    step=global_step,
                )

            # ===== save & dev eval =====
            if epoch >= 0:
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(args.output_dir)
                processor.save_pretrained(args.output_dir)
                print(f"[train] epoch {epoch+1} saved adapter to: {args.output_dir}")

                if dev_loader is not None:
                    dev_metrics = run_dev_eval(
                        model=model,
                        processor=processor,
                        dev_loader=dev_loader,
                        output_dir=args.output_dir,
                        epoch_idx=epoch,
                        use_amp=use_amp,
                        autocast_dtype=autocast_dtype,
                        eval_max_new_tokens=args.eval_max_new_tokens,
                        iou_threshold=args.eval_iou,
                        strict_gold=(not args.no_strict_gold),
                        log_file=eval_log_path,
                    )

                    cur = float(dev_metrics.get("dev/gmner/f1", -1.0))
                    if cur > best_dev_score:
                        best_dev_score = cur
                        best_epoch = epoch + 1

                        os.makedirs(best_dir, exist_ok=True)
                        model.save_pretrained(best_dir)
                        processor.save_pretrained(best_dir)

                        if get_peft_model_state_dict is not None:
                            sd = get_peft_model_state_dict(model)
                            best_state = {k: v.detach().cpu().clone() for k, v in sd.items()}
                        else:
                            best_state = None

                        print(f"[best] update: epoch={best_epoch} dev_gmner_f1={best_dev_score:.6f} -> saved to {best_dir}")
                        append_text_log(
                            eval_log_path,
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BEST update epoch={best_epoch} dev_gmner_f1={best_dev_score:.6f} saved_dir={best_dir}",
                        )

                    if swan_run is not None and isinstance(dev_metrics, dict):
                        dev_log = {
                            "dev/loss": float(dev_metrics["dev/loss"]),
                            "dev/gmner/f1": float(dev_metrics["dev/gmner/f1"]),
                            "dev/mner/f1": float(dev_metrics["dev/mner/f1"]),
                            "dev/eeg/f1": float(dev_metrics["dev/eeg/f1"]),
                        }
                        _swan_log(swan_run, dev_log, step=global_step)

        print(f"Training finished. Saved adapter to: {args.output_dir}")

        # ===== 训练结束：用 best 跑 test，并写入 eval_results.txt =====
        if test_loader is not None:
            if dev_loader is None:
                print("[test] dev_jsonl is None -> evaluate test with LAST checkpoint (current model weights)")
                run_test_eval(
                    model=model,
                    processor=processor,
                    test_loader=test_loader,
                    output_dir=args.output_dir,
                    tag="last",
                    use_amp=use_amp,
                    autocast_dtype=autocast_dtype,
                    eval_max_new_tokens=args.eval_max_new_tokens,
                    iou_threshold=args.eval_iou,
                    strict_gold=(not args.no_strict_gold),
                    log_file=eval_log_path,
                )
            else:
                print(f"[best] final best_epoch={best_epoch} best_dev_gmner_f1={best_dev_score:.6f}")
                append_text_log(
                    eval_log_path,
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BEST final epoch={best_epoch} dev_gmner_f1={best_dev_score:.6f}",
                )

                restored = False
                if best_state is not None and set_peft_model_state_dict is not None:
                    set_peft_model_state_dict(model, best_state)
                    restored = True
                    print("[best] restored best LoRA from in-memory state_dict")
                    append_text_log(
                        eval_log_path,
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BEST restored from state_dict",
                    )
                else:
                    if hasattr(model, "load_adapter"):
                        try:
                            model.load_adapter(best_dir, adapter_name="best", is_trainable=False)
                            if hasattr(model, "set_adapter"):
                                model.set_adapter("best")
                            restored = True
                            print(f"[best] restored best LoRA from folder: {best_dir}")
                            append_text_log(
                                eval_log_path,
                                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] BEST restored from folder {best_dir}",
                            )
                        except Exception as e:
                            print(f"[WARN] load_adapter failed: {e}")
                            append_text_log(
                                eval_log_path,
                                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARN load_adapter failed: {e}",
                            )

                if not restored:
                    print(f"[WARN] cannot restore best adapter automatically. You can manually load adapter from: {best_dir}")
                    append_text_log(
                        eval_log_path,
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARN cannot restore best adapter automatically; best_dir={best_dir}",
                    )

                run_test_eval(
                    model=model,
                    processor=processor,
                    test_loader=test_loader,
                    output_dir=args.output_dir,
                    tag=f"best_ep{best_epoch}",
                    use_amp=use_amp,
                    autocast_dtype=autocast_dtype,
                    eval_max_new_tokens=args.eval_max_new_tokens,
                    iou_threshold=args.eval_iou,
                    strict_gold=(not args.no_strict_gold),
                    log_file=eval_log_path,
                )

        append_text_log(eval_log_path, f"========== Run end {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========")

    finally:
        _swan_finish(swan_run)
