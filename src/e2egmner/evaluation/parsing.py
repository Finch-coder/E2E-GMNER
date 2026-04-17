"""Parsing helpers for structured <answer> outputs and bbox triples."""

import json
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

FULLWIDTH_BAR_CANON = "|"  # ASCII
BAR_VARIANTS = ["｜", "│", "丨", "￨", "∣"] #可能错误的分割字符
ENTITY_SPLIT_RE = re.compile(r"[;；\n]+")

Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class EntityTriple:
    text: str
    etype: str
    regions: Optional[List[Box]]
    region_valid: bool = True


def normalize_separators(s: str) -> str:
    if s is None:
        return ""
    for b in BAR_VARIANTS:
        s = s.replace(b, FULLWIDTH_BAR_CANON)
    return s


# ==========================
# ✅ CoT/Answer 提取与替换
# ==========================
THINK_TAG_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)


def extract_answer_text(s: Optional[str]) -> str:
    """从模型输出中抽取用于解析/评测/缩放 bbox 的 answer 文本。"""
    if not s:
        return ""
    s = s.strip()
    #提取answer中的三元组答案：Hailey Martin|PER|[201,81,396,225]
    m = ANSWER_TAG_RE.search(s)
    if m:
        return (m.group(1) or "").strip()

    # 兼容：只有 </think> 没有 <answer>
    m2 = re.search(r"</think>\s*(.*)$", s, flags=re.S | re.I)
    if m2:
        return (m2.group(1) or "").strip()

    # 兼容：本来就是纯结构化输出
    return s


def replace_answer_text(full: Optional[str], new_answer: str) -> str:
    """把 full 文本里的 <answer>...</answer> 内容替换为 new_answer；若不存在则尽量补齐。"""
    new_answer = (new_answer or "").strip()
    if not full:
        return new_answer

    s = full.strip()

    if ANSWER_TAG_RE.search(s):
        return ANSWER_TAG_RE.sub(lambda m: f"<answer>{new_answer}</answer>", s, count=1)

    # 有 think 但没有 answer：把 answer 补到 think 后面
    if re.search(r"</think>", s, flags=re.I):
        return re.sub(r"(</think>\s*).*$", r"\1" + f"<answer>{new_answer}</answer>", s, flags=re.S | re.I)

    # 啥标签都没有：直接返回 answer
    return new_answer


def _boxes_from_numbers(nums: List[float], *, strict: bool, where: str) -> List[Box]:
    if len(nums) % 4 != 0:
        msg = f"[{where}] bbox numbers count not multiple of 4: {len(nums)} -> {nums}"
        if strict:
            raise ValueError(msg)
        return []
    boxes: List[Box] = []
    for i in range(0, len(nums), 4):
        x1, y1, x2, y2 = nums[i:i+4]
        boxes.append((x1, y1, x2, y2))
    return boxes


def parse_bbox_regions(bbox_str: str, *, strict: bool, where: str) -> Tuple[Optional[List[Box]], bool]:
    if bbox_str is None:
        return None, True

    s = bbox_str.strip()
    if not s:
        if strict:
            raise ValueError(f"[{where}] empty bbox string")
        return None, False

    low = s.lower()
    if low in {"none", "n/a"}:
        return None, True

    try:
        js = json.loads(s)
        if isinstance(js, list) and len(js) == 4 and all(isinstance(v, (int, float)) for v in js):
            return [(float(js[0]), float(js[1]), float(js[2]), float(js[3]))], True
        if isinstance(js, list) and js and all(isinstance(b, list) for b in js):
            boxes: List[Box] = []
            for b in js:
                if len(b) != 4 or not all(isinstance(v, (int, float)) for v in b):
                    raise ValueError(f"[{where}] invalid box element: {b}")
                boxes.append((float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            return boxes, True
        if isinstance(js, list) and len(js) == 0:
            if strict:
                raise ValueError(f"[{where}] empty list bbox is invalid")
            return None, False
    except Exception:
        pass

    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        if strict:
            raise ValueError(f"[{where}] cannot parse bbox: {bbox_str!r}")
        return None, False

    vals = [float(x) for x in nums]
    boxes = _boxes_from_numbers(vals, strict=strict, where=where)
    if not boxes:
        if strict:
            raise ValueError(f"[{where}] invalid bbox numbers: {bbox_str!r}")
        return None, False
    return boxes, True


def parse_triple_segment(seg: str, *, strict: bool, where: str) -> Optional[EntityTriple]:
    seg = seg.strip()
    if not seg or seg.lower() == "none":
        return None
    #这里是处理 模型输出的不同格式'|'的情况
    seg = normalize_separators(seg)
    # if FULLWIDTH_BAR_CANON in seg:
    #     left, right = seg.split(FULLWIDTH_BAR_CANON, 1)
    #     left = left.strip()
    #     right = right.strip()

    #     if "|" not in left:
    #         if strict:
    #             raise ValueError(f"[{where}] missing '|' in left part: {seg!r}")
    #         return None

    #     mention_part, type_part = left.rsplit("|", 1)
    #     mention = mention_part.strip()
    #     etype = type_part.strip()

    #     if not mention or not etype:
    #         if strict:
    #             raise ValueError(f"[{where}] empty mention/type: {seg!r}")
    #         return None

    #     regions, valid = parse_bbox_regions(right, strict=strict, where=where)
    #     return EntityTriple(text=mention, etype=etype, regions=regions, region_valid=valid)
    
    parts = [p.strip() for p in seg.split("|") if p.strip()]
    if len(parts) < 3:
        if strict:
            raise ValueError(f"[{where}] cannot parse segment (too few parts): {seg!r}")
        return None

    mention = " | ".join(parts[:-2]).strip()#兼容 mention 里本身带 | 的情况 
    etype = parts[-2].strip()
    bbox_str = parts[-1].strip()

    if not mention or not etype:
        if strict:
            raise ValueError(f"[{where}] empty mention/type in degraded segment: {seg!r}")
        return None

    regions, valid = parse_bbox_regions(bbox_str, strict=strict, where=where)
    return EntityTriple(text=mention, etype=etype, regions=regions, region_valid=valid)


def parse_triples(s: str, *, strict: bool, where: str) -> List[EntityTriple]:
    if s is None:
        return []
    s = normalize_separators(s).strip()
    if not s or s.lower() == "none":
        return []
    
   
    segments = [seg.strip() for seg in ENTITY_SPLIT_RE.split(s) if seg.strip()] #将多个实体三元组分割成列表形式
    triples: List[EntityTriple] = []
    for seg in segments:
        if seg.lower() == "none":
            continue
        triple = parse_triple_segment(seg, strict=strict, where=where)
        if triple is not None:
            triples.append(triple)
    return triples

def _short_text_for_log(s: Optional[str], limit: int = 120) -> str:
    s = (s or '').replace('\n', ' ').strip()
    if len(s) <= limit:
        return s
    return s[: limit - 3] + '...'


def _structured_answer_parse_failed(ans: str) -> bool:
    ans = (ans or '').strip()
    if not ans or ans.lower() == 'none':
        return False
    # 本任务的 answer 期望是结构化实体三元组；只要非空但解析不出 triples，就视作失败，避免静默错位。
    return True


def triples_to_canon_text(triples: List[EntityTriple]) -> str:
    segs = []
    for t in triples:
        if t.regions is None:
            bbox_str = "None"
        else:
            boxes = []
            for (x1, y1, x2, y2) in t.regions:
                boxes.append([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))])
            if len(boxes) == 1:
                bbox_str = "[" + ",".join(map(str, boxes[0])) + "]"
            else:
                bbox_str = json.dumps(boxes, ensure_ascii=False, separators=(",", ":"))
        segs.append(f"{t.text}|{t.etype}|{bbox_str}")
    return "; ".join(segs)
