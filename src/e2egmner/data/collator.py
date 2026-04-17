"""Chat-template collator used for supervised fine-tuning and evaluation."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..evaluation.geometry import rescale_assistant_text_bboxes

DEFAULT_SYSTEM_PROMPT = (
    "You are an information extraction assistant for grounded multimodal named entity recognition. "
    "Given the image and text, first think briefly and then answer in a strict XML-like format. "
    "The final output must contain exactly one <think>...</think> block followed by exactly one "
    "<answer>...</answer> block. "
    "Inside <answer>, output only structured entity triples in the format "
    "mention|type|[x1,y1,x2,y2], separated by '; '. "
    "If there is no entity, output <answer>None</answer>. "
    "Do not add any extra explanation outside the tags."
)

class Qwen25VLSFTCollator:
    def __init__(
        self,
        processor,
        max_length: int = 2048,
        *,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        factor: int = 28,
        do_jitter: bool = False,
        jitter_beta: float = 0.05,
        jitter_gamma: float = 0.05,
        jitter_iou_min: float = 0.6,
        jitter_tries: int = 3,
        jitter_min_area: int = 32 * 32,
        jitter_small_box_scale: float = 0.5,
        jitter_dist: str = "gauss",
        jitter_gauss_trunc_k: float = 2.0,
        jitter_scale_min: float = 0.2,
        jitter_scale_max: float = 5.0,
        system_prompt: Optional[str] = None,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.system_prompt = (system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT).strip()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"

        ip = getattr(processor, "image_processor", None)
        self.min_pixels = int(min_pixels) if min_pixels is not None else int(getattr(ip, "min_pixels", 256 * 28 * 28))
        self.max_pixels = int(max_pixels) if max_pixels is not None else int(getattr(ip, "max_pixels", 1024 * 28 * 28))
        self.factor = int(factor)

        # jitter: 仅训练时建议开启；dev/test 保持关闭
        self.do_jitter = bool(do_jitter)
        self.jitter_beta = float(jitter_beta)
        self.jitter_gamma = float(jitter_gamma)
        self.jitter_iou_min = float(jitter_iou_min)
        self.jitter_tries = int(jitter_tries)
        self.jitter_min_area = int(jitter_min_area)
        self.jitter_small_box_scale = float(jitter_small_box_scale)
        self.jitter_dist = str(jitter_dist)
        self.jitter_gauss_trunc_k = float(jitter_gauss_trunc_k)
        self.jitter_scale_min = float(jitter_scale_min)
        self.jitter_scale_max = float(jitter_scale_max)

    def build_messages(self, user_text: str, assistant_text: Optional[str] = None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        user_msg = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
        messages.append(user_msg)

        if assistant_text is not None:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    def __call__(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        images = []
        scaled_assistant_texts: List[str] = []
        orig_assistant_texts: List[str] = []
        orig_sizes: List[Tuple[int, int]] = []
        valid_batch: List[Dict[str, Any]] = []

        # 先过滤 batch：跳过 smart_resize 出错 / 返回 None 的样本
        for x in batch:
            img = x["image"]
            orig_w, orig_h = img.size

            try:
                scaled_a = rescale_assistant_text_bboxes(
                    x["assistant_text"],
                    orig_w=orig_w,
                    orig_h=orig_h,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                    factor=self.factor,
                    do_jitter=self.do_jitter,
                    jitter_beta=self.jitter_beta,
                    jitter_gamma=self.jitter_gamma,
                    jitter_iou_min=self.jitter_iou_min,
                    jitter_tries=self.jitter_tries,
                    jitter_min_area=self.jitter_min_area,
                    jitter_small_box_scale=self.jitter_small_box_scale,
                    jitter_dist=self.jitter_dist,
                    jitter_gauss_trunc_k=self.jitter_gauss_trunc_k,
                    jitter_scale_min=self.jitter_scale_min,
                    jitter_scale_max=self.jitter_scale_max,
                )
            except ValueError as e:
                print(f"[WARNING] skipping sample due to resize error: {e}")
                continue

            if scaled_a is None:
                continue

            valid_batch.append(x)
            images.append(img)
            scaled_assistant_texts.append(scaled_a)
            orig_assistant_texts.append(x["assistant_text"])
            orig_sizes.append((orig_w, orig_h))

        if not valid_batch:
            # 让上层训练/评测循环显式跳过空 batch，避免把“假样本”送进模型
            return None

        prompt_texts: List[str] = []
        full_texts: List[str] = []

        for x, a_scaled in zip(valid_batch, scaled_assistant_texts):
            u = x["user_text"]
            prompt_messages = self.build_messages(u, assistant_text=None)
            full_messages = self.build_messages(u, assistant_text=a_scaled)

            prompt = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            full = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )
            prompt_texts.append(prompt)
            full_texts.append(full)

        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        full_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        self.tokenizer.padding_side = "left" #prompt_inputs用来推理 所以左padding
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        self.tokenizer.padding_side = old_side

        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs["attention_mask"]
        prompt_lens = prompt_inputs["attention_mask"].sum(dim=1)

        labels = input_ids.clone()
        labels[:] = -100
        for i in range(input_ids.size(0)):
            p_len = int(prompt_lens[i].item())
            labels[i, p_len:] = input_ids[i, p_len:] 
            labels[i, attention_mask[i] == 0] = -100

        full_inputs["labels"] = labels
        full_inputs["prompt_input_ids"] = prompt_inputs["input_ids"]
        full_inputs["prompt_attention_mask"] = prompt_inputs["attention_mask"]
        full_inputs["gold_texts"] = scaled_assistant_texts      # 训练/评测用：缩放后的 gold
        full_inputs["gold_texts_orig"] = orig_assistant_texts   # 导出/恢复用：原图坐标 gold
        full_inputs["orig_sizes"] = orig_sizes                  # 记录原图尺寸

        return full_inputs
