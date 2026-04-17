"""Evaluation loops for dev and test sets."""

import json
import os
from datetime import datetime
from typing import Optional

import torch
from tqdm import tqdm

from ..evaluation.geometry import restore_assistant_text_bboxes_to_original
from ..evaluation.metrics import count_correct_eeg, count_correct_gmner, count_correct_mner, prf
from ..evaluation.parsing import extract_answer_text, parse_triples
from ..utils.runtime import append_text_log, get_first_param_device, pick_vision_inputs

@torch.no_grad()
def run_dev_eval(
    model,
    processor,
    dev_loader,
    output_dir: str,
    epoch_idx: int,
    use_amp: bool,
    autocast_dtype,
    eval_max_new_tokens: int,
    iou_threshold: float = 0.5,
    strict_gold: bool = True,
    log_file: Optional[str] = None,
):
    model.eval()
    device = get_first_param_device(model)
    collator = dev_loader.collate_fn

    total_loss = 0.0
    total_tokens = 0

    gmner_correct = gmner_pred = gmner_gold = 0
    mner_correct = mner_pred = mner_gold = 0
    eeg_correct = eeg_pred = eeg_gold = 0
    bad_pred_bbox = 0
    bad_restore_parse = 0

    pred_path = os.path.join(output_dir, f"dev_pred_epoch{epoch_idx+1}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    with open(pred_path, "w", encoding="utf-8") as wf:
        for batch in tqdm(dev_loader, desc=f"[dev] epoch {epoch_idx+1}", leave=False):
            if batch is None:
                continue

            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            vision_inputs = pick_vision_inputs(batch)

            # ---- dev loss ----
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    **vision_inputs,
                )
                loss = out.loss

            valid_tokens = int((batch["labels"] != -100).sum().item())
            if valid_tokens > 0:
                total_loss += float(loss.item()) * valid_tokens
                total_tokens += valid_tokens

            # ---- generate ----
            gen_inputs = {
                "input_ids": batch["prompt_input_ids"],
                "attention_mask": batch["prompt_attention_mask"],
                **vision_inputs,
            }
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
                generated_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=eval_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                )

            trimmed = [
                out_ids[in_ids.shape[-1]:] for in_ids, out_ids in zip(gen_inputs["input_ids"], generated_ids)
            ]
            pred_texts = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            gold_texts = batch["gold_texts"]
            gold_texts_orig = batch.get("gold_texts_orig", gold_texts)
            orig_sizes = batch["orig_sizes"]

            for i, (pred, gold_scaled, gold_orig, (orig_w, orig_h)) in enumerate(
                zip(pred_texts, gold_texts, gold_texts_orig, orig_sizes)
            ):
                pred = (pred or "").strip()
                gold_scaled = (gold_scaled or "").strip()
                gold_orig = (gold_orig or "").strip()

                pred_ans_scaled = extract_answer_text(pred)
                gold_ans_scaled = extract_answer_text(gold_scaled)

                pred_restored_full = restore_assistant_text_bboxes_to_original(
                    pred,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    min_pixels=collator.min_pixels,
                    max_pixels=collator.max_pixels,
                    factor=collator.factor,
                )
                if pred_restored_full is None and pred_ans_scaled and pred_ans_scaled.lower() != "none":
                    bad_restore_parse += 1
                pred_restored_full = (pred_restored_full or "").strip()
                pred_ans_orig = extract_answer_text(pred_restored_full)
                gold_ans_orig = extract_answer_text(gold_orig)

                wf.write(
                    json.dumps(
                        {
                            "response_full": pred,
                            "response_full_orig": pred_restored_full,
                            "response_answer_scaled": pred_ans_scaled,
                            "response_answer_orig": pred_ans_orig,
                            "labels_full_scaled": gold_scaled,
                            "labels_full_orig": gold_orig,
                            "labels_answer_scaled": gold_ans_scaled,
                            "labels_answer_orig": gold_ans_orig,
                            "orig_size": [orig_w, orig_h],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # ✅ 指标仍然按 scaled 坐标算，保持和现有逻辑一致
                pred_triples = parse_triples(pred_ans_scaled, strict=False, where=f"pred epoch{epoch_idx+1} idx{i}")
                gold_triples = parse_triples(gold_ans_scaled, strict=strict_gold, where=f"gold epoch{epoch_idx+1} idx{i}")

                bad_pred_bbox += sum(1 for t in pred_triples if not t.region_valid)

                gmner_pred += len(pred_triples)
                gmner_gold += len(gold_triples)
                gmner_correct += count_correct_gmner(pred_triples, gold_triples, iou_threshold)

                mner_pred += len(pred_triples)
                mner_gold += len(gold_triples)
                mner_correct += count_correct_mner(pred_triples, gold_triples)

                eeg_pred += len(pred_triples)
                eeg_gold += len(gold_triples)
                eeg_correct += count_correct_eeg(pred_triples, gold_triples, iou_threshold)

    dev_loss = (total_loss / total_tokens) if total_tokens > 0 else float("nan")

    gmner_p, gmner_r, gmner_f1 = prf(gmner_correct, gmner_pred, gmner_gold)
    mner_p, mner_r, mner_f1 = prf(mner_correct, mner_pred, mner_gold)
    eeg_p, eeg_r, eeg_f1 = prf(eeg_correct, eeg_pred, eeg_gold)

    print(f"\n[dev] epoch={epoch_idx+1} dev_loss={dev_loss:.6f}  (token-weighted)")
    print(f"[dev] saved dev_pred: {pred_path}")
    if bad_pred_bbox:
        print(f"[dev] [INFO] Invalid pred bbox segments (won't match region): {bad_pred_bbox}")
    if bad_restore_parse:
        print(f"[dev] [INFO] Restore-to-original parse failures: {bad_restore_parse}")

    print("========== DEV GMNER (entity + type + region) ==========")
    print(f"correct={gmner_correct}, pred={gmner_pred}, gold={gmner_gold}")
    print(f"P={gmner_p:.4f}  R={gmner_r:.4f}  F1={gmner_f1:.4f}")

    print("========== DEV MNER (entity + type) ==========")
    print(f"correct={mner_correct}, pred={mner_pred}, gold={mner_gold}")
    print(f"P={mner_p:.4f}  R={mner_r:.4f}  F1={mner_f1:.4f}")

    print("========== DEV EEG (entity + region) ==========")
    print(f"correct={eeg_correct}, pred={eeg_pred}, gold={eeg_gold}")
    print(f"P={eeg_p:.4f}  R={eeg_r:.4f}  F1={eeg_f1:.4f}\n")

    if log_file is not None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"[{ts}] DEV epoch={epoch_idx+1} "
            f"loss={dev_loss:.6f} "
            f"gmner_f1={gmner_f1:.4f} (P={gmner_p:.4f},R={gmner_r:.4f}) "
            f"mner_f1={mner_f1:.4f} (P={mner_p:.4f},R={mner_r:.4f}) "
            f"eeg_f1={eeg_f1:.4f} (P={eeg_p:.4f},R={eeg_r:.4f}) "
            f"bad_bbox={bad_pred_bbox} "
            f"bad_restore_parse={bad_restore_parse} "
            f"pred_file={os.path.basename(pred_path)}"
        )
        append_text_log(log_file, line)

    model.train()

    return {
        "dev/loss": dev_loss,
        "dev/gmner/p": gmner_p,
        "dev/gmner/r": gmner_r,
        "dev/gmner/f1": gmner_f1,
        "dev/mner/p": mner_p,
        "dev/mner/r": mner_r,
        "dev/mner/f1": mner_f1,
        "dev/eeg/p": eeg_p,
        "dev/eeg/r": eeg_r,
        "dev/eeg/f1": eeg_f1,
        "dev/bad_pred_bbox": bad_pred_bbox,
        "dev/bad_restore_parse": bad_restore_parse,
        "dev/gmner_pred": gmner_pred,
        "dev/gmner_gold": gmner_gold,
        "dev/mner_pred": mner_pred,
        "dev/mner_gold": mner_gold,
        "dev/eeg_pred": eeg_pred,
        "dev/eeg_gold": eeg_gold,
        "dev/pred_path": pred_path,
    }


# ============================================================
# test eval: generate + metrics + save jsonl
# ============================================================
@torch.no_grad()
def run_test_eval(
    model,
    processor,
    test_loader,
    output_dir: str,
    tag: str,
    use_amp: bool,
    autocast_dtype,
    eval_max_new_tokens: int,
    iou_threshold: float = 0.5,
    strict_gold: bool = True,
    log_file: Optional[str] = None,
):
    
    model.eval()
    device = get_first_param_device(model)
    collator = test_loader.collate_fn

    gmner_correct = gmner_pred = gmner_gold = 0
    mner_correct = mner_pred = mner_gold = 0
    eeg_correct = eeg_pred = eeg_gold = 0
    bad_pred_bbox = 0
    bad_restore_parse = 0

    pred_path = os.path.join(output_dir, f"test_pred_{tag}.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    with open(pred_path, "w", encoding="utf-8") as wf:
        for batch in tqdm(test_loader, desc=f"[test] {tag}", leave=False):
            if batch is None:
                continue

            for k, v in list(batch.items()):
                if torch.is_tensor(v):
                    batch[k] = v.to(device, non_blocking=True)

            vision_inputs = pick_vision_inputs(batch)

            gen_inputs = {
                "input_ids": batch["prompt_input_ids"],
                "attention_mask": batch["prompt_attention_mask"],
                **vision_inputs,
            }

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=autocast_dtype):
                generated_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=eval_max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                )

            trimmed = [
                out_ids[in_ids.shape[-1]:] for in_ids, out_ids in zip(gen_inputs["input_ids"], generated_ids) #切掉prompt前缀 只取 completion
            ]
            pred_texts = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            gold_texts = batch["gold_texts"]
            gold_texts_orig = batch.get("gold_texts_orig", gold_texts)
            orig_sizes = batch["orig_sizes"]

            for i, (pred, gold_scaled, gold_orig, (orig_w, orig_h)) in enumerate(
                zip(pred_texts, gold_texts, gold_texts_orig, orig_sizes)
            ):
                pred = (pred or "").strip()
                gold_scaled = (gold_scaled or "").strip()#缩放后的图像尺寸
                gold_orig = (gold_orig or "").strip()#没缩放的

                # 1) scaled（当前评测仍然用它）
                pred_ans_scaled = extract_answer_text(pred)
                gold_ans_scaled = extract_answer_text(gold_scaled)

                # 2) orig
                pred_restored_full = restore_assistant_text_bboxes_to_original(
                    pred,
                    orig_w=orig_w,
                    orig_h=orig_h,
                    min_pixels=collator.min_pixels,
                    max_pixels=collator.max_pixels,
                    factor=collator.factor,
                )
                if pred_restored_full is None and pred_ans_scaled and pred_ans_scaled.lower() != "none":
                    bad_restore_parse += 1
                pred_restored_full = (pred_restored_full or "").strip()
                pred_ans_orig = extract_answer_text(pred_restored_full)
                gold_ans_orig = extract_answer_text(gold_orig)

                wf.write(
                    json.dumps(
                        {
                            "response_full": pred,
                            "response_full_orig": pred_restored_full,
                            "response_answer_scaled": pred_ans_scaled,
                            "response_answer_orig": pred_ans_orig,
                            "labels_full_scaled": gold_scaled,
                            "labels_full_orig": gold_orig,
                            "labels_answer_scaled": gold_ans_scaled,
                            "labels_answer_orig": gold_ans_orig,
                            "orig_size": [orig_w, orig_h],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                # 指标继续按 scaled 算，保证和现在一致
                pred_triples = parse_triples(pred_ans_scaled, strict=False, where=f"pred test {tag} idx{i}")
                gold_triples = parse_triples(gold_ans_scaled, strict=strict_gold, where=f"gold test {tag} idx{i}")

                bad_pred_bbox += sum(1 for t in pred_triples if not t.region_valid)

                gmner_pred += len(pred_triples)
                gmner_gold += len(gold_triples)
                gmner_correct += count_correct_gmner(pred_triples, gold_triples, iou_threshold)

                mner_pred += len(pred_triples)
                mner_gold += len(gold_triples)
                mner_correct += count_correct_mner(pred_triples, gold_triples)

                eeg_pred += len(pred_triples)
                eeg_gold += len(gold_triples)
                eeg_correct += count_correct_eeg(pred_triples, gold_triples, iou_threshold)

    gmner_p, gmner_r, gmner_f1 = prf(gmner_correct, gmner_pred, gmner_gold)
    mner_p, mner_r, mner_f1 = prf(mner_correct, mner_pred, mner_gold)
    eeg_p, eeg_r, eeg_f1 = prf(eeg_correct, eeg_pred, eeg_gold)

    print(f"\n[test] tag={tag}  saved: {pred_path}")
    if bad_pred_bbox:
        print(f"[test] [INFO] Invalid pred bbox segments: {bad_pred_bbox}")
    if bad_restore_parse:
        print(f"[test] [INFO] Restore-to-original parse failures: {bad_restore_parse}")

    print("========== TEST GMNER (entity + type + region) ==========")
    print(f"correct={gmner_correct}, pred={gmner_pred}, gold={gmner_gold}")
    print(f"P={gmner_p:.4f}  R={gmner_r:.4f}  F1={gmner_f1:.4f}")

    print("========== TEST MNER (entity + type) ==========")
    print(f"correct={mner_correct}, pred={mner_pred}, gold={mner_gold}")
    print(f"P={mner_p:.4f}  R={mner_r:.4f}  F1={mner_f1:.4f}")

    print("========== TEST EEG (entity + region) ==========")
    print(f"correct={eeg_correct}, pred={eeg_pred}, gold={eeg_gold}")
    print(f"P={eeg_p:.4f}  R={eeg_r:.4f}  F1={eeg_f1:.4f}\n")

    if log_file is not None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"[{ts}] TEST tag={tag} "
            f"gmner_f1={gmner_f1:.4f} (P={gmner_p:.4f},R={gmner_r:.4f}) "
            f"mner_f1={mner_f1:.4f} (P={mner_p:.4f},R={mner_r:.4f}) "
            f"eeg_f1={eeg_f1:.4f} (P={eeg_p:.4f},R={eeg_r:.4f}) "
            f"bad_bbox={bad_pred_bbox} "
            f"bad_restore_parse={bad_restore_parse} "
            f"pred_file={os.path.basename(pred_path)}"
        )
        append_text_log(log_file, line)

    model.train()

    return {
        "test/gmner/p": gmner_p,
        "test/gmner/r": gmner_r,
        "test/gmner/f1": gmner_f1,
        "test/mner/p": mner_p,
        "test/mner/r": mner_r,
        "test/mner/f1": mner_f1,
        "test/eeg/p": eeg_p,
        "test/eeg/r": eeg_r,
        "test/eeg/f1": eeg_f1,
        "test/bad_pred_bbox": bad_pred_bbox,
        "test/bad_restore_parse": bad_restore_parse,
        "test/pred_path": pred_path,
    }
