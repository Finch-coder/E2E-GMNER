"""Dataset helpers for grounded multimodal NER."""

import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

#兼容旧格式数据
def normalize_user_text(s: str) -> str:
    s = s.replace("<image> Text:", "<image>\nText:")
    return s


def strip_image_token_and_prefix(s: str) -> str:
    s = normalize_user_text(s)
    s = s.replace("<image>", "").strip()
    return s


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


class TwitterGroundedMNERJsonl(Dataset):
    def __init__(self, jsonl_path: str, image_root: Optional[str] = None):
        self.items: List[Dict[str, Any]] = []
        self.image_root = image_root
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        msgs = ex["messages"]

        user_text = strip_image_token_and_prefix(msgs[0]["content"])
        assistant_text = msgs[-1]["content"]

        img_path = ex["images"][0]
        if self.image_root is not None and (not os.path.isabs(img_path)):
            img_path = os.path.join(self.image_root, img_path)
        image = Image.open(img_path).convert("RGB")

        return {"user_text": user_text, "assistant_text": assistant_text, "image": image}
