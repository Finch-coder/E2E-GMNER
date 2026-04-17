"""Geometry, resizing, jittering, and bbox rescaling utilities."""

import math
import random
from typing import List, Optional, Tuple

from .parsing import (
    Box,
    EntityTriple,
    _short_text_for_log,
    _structured_answer_parse_failed,
    extract_answer_text,
    parse_triples,
    replace_answer_text,
    triples_to_canon_text,
)

def iou(box1: Box, box2: Box) -> float:
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    ix1 = max(x1, x1g)
    iy1 = max(y1, y1g)
    ix2 = min(x2, x2g)
    iy2 = min(y2, y2g)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x2g - x1g) * max(0.0, y2g - y1g)
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union



def max_iou(pred_boxes: List[Box], gold_boxes: List[Box]) -> float:
    """
这里遵从第一篇GMNER论文的评测多框的准则方法
Grounded Multimodal Named Entity Recognition on Social Media  Jianfei Yu∗, Ziyan Li∗, Jieming Wang and Rui Xia†


For visual region, if it is ungroundable, the prediction is considered as correct only when it is None; otherwise, the prediction is considered as correct only when the 
IoU score between the predicted visual region and one of the groundtruth (GT) bounding boxes is large than 0.5

面对多框问题，只要预测的框与多框中的一个框IOU>0.5即算预测对
"""
    best = 0.0
    for pb in pred_boxes:
        for gb in gold_boxes:
            val = iou(pb, gb)
            if val > best:
                best = val
    return best


def region_correct(p: EntityTriple, g: EntityTriple, iou_threshold: float) -> Tuple[bool, float]:
    """
    这里遵从第一篇GMNER论文的评测多框的准则方法
    Grounded Multimodal Named Entity Recognition on Social Media  Jianfei Yu∗, Ziyan Li∗, Jieming Wang and Rui Xia†


    For visual region, if it is ungroundable, the prediction is considered as correct only when it is None; otherwise, the prediction is considered as correct only when the 
    IoU score between the predicted visual region and one of the groundtruth (GT) bounding boxes is large than 0.5

    面对多框问题，只要预测的框与多框中的一个框IOU>0.5即算预测对
    """

    if not p.region_valid:
        return False, 0.0

    if p.regions is None and g.regions is None:
        return True, 1.0
    if p.regions is None or g.regions is None:
        return False, 0.0

    miou = max_iou(p.regions, g.regions)
    if miou >= iou_threshold:
        return True, miou
    return False, miou

def _clip_box_xyxy(box: Box, W: int, H: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(W), x1))
    x2 = max(0.0, min(float(W), x2))
    y1 = max(0.0, min(float(H), y1))
    y2 = max(0.0, min(float(H), y2))
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def _box_wh(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _sample_jitter_delta(
    a: float,
    *,
    dist: str,
    gauss_sigma: Optional[float] = None,
    gauss_trunc_k: float = 2.0,
) -> float:
    """
    返回扰动值（相对比例，不乘 w/h）
    - uniform: U(-a, a)
    - gauss: N(0, sigma) 然后 clip 到 [-cap, cap]
      sigma 默认取 a/sqrt(3) 使其与 uniform 方差接近
      cap = min(a, k*sigma)
    """
    a = float(a)
    if a <= 0:
        return 0.0

    dist = (dist or "gauss").lower()
    if dist in {"uniform", "u"}:
        return random.uniform(-a, a)

    # gauss
    if gauss_sigma is None:
        gauss_sigma = a / math.sqrt(3.0)
    sigma = max(1e-12, float(gauss_sigma))
    cap = min(a, float(gauss_trunc_k) * sigma) if gauss_trunc_k is not None else a
    v = random.gauss(0.0, sigma)
    v = _clamp(v, -cap, cap)
    return v


def jitter_box_center_scale(
    gt: Box,
    W: int,
    H: int,
    beta: float,
    gamma: float,
    min_size: float = 2.0,
    *,
    jitter_dist: str = "gauss",
    gauss_trunc_k: float = 2.0,
    scale_min: float = 0.2,
    scale_max: float = 5.0,
) -> Box:
    x1, y1, x2, y2 = gt
    w, h = _box_wh(gt)
    if w < min_size or h < min_size:
        return gt

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # center shift
    dx_r = _sample_jitter_delta(beta, dist=jitter_dist, gauss_trunc_k=gauss_trunc_k)
    dy_r = _sample_jitter_delta(beta, dist=jitter_dist, gauss_trunc_k=gauss_trunc_k)
    dx = dx_r * w
    dy = dy_r * h

    # scale
    sx_r = _sample_jitter_delta(gamma, dist=jitter_dist, gauss_trunc_k=gauss_trunc_k)
    sy_r = _sample_jitter_delta(gamma, dist=jitter_dist, gauss_trunc_k=gauss_trunc_k)
    sx = _clamp(1.0 + sx_r, float(scale_min), float(scale_max))
    sy = _clamp(1.0 + sy_r, float(scale_min), float(scale_max))

    w2 = max(min_size, w * sx)
    h2 = max(min_size, h * sy)

    cx2 = cx + dx
    cy2 = cy + dy

    j = (cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0)
    j = _clip_box_xyxy(j, W, H)

    wj, hj = _box_wh(j)
    if wj < min_size or hj < min_size:
        return gt
    return j


def jitter_box_with_iou_guard(
    gt: Box,
    W: int,
    H: int,
    beta: float,
    gamma: float,
    iou_min: float = 0.6,
    tries: int = 3,
    min_size: float = 2.0,
    *,
    jitter_dist: str = "gauss",
    gauss_trunc_k: float = 2.0,
    scale_min: float = 0.2,
    scale_max: float = 5.0,
) -> Box:
    gt = _clip_box_xyxy(gt, W, H)
    for _ in range(max(1, tries)):
        j = jitter_box_center_scale(
            gt,
            W,
            H,
            beta=beta,
            gamma=gamma,
            min_size=min_size,
            jitter_dist=jitter_dist,
            gauss_trunc_k=gauss_trunc_k,
            scale_min=scale_min,
            scale_max=scale_max,
        )
        if iou(j, gt) >= iou_min:
            return j
    return gt


# ============================================================
# ✅ 动态分辨率：bbox 同步缩放工具
# ============================================================
def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> Tuple[int, int]:
    """
    与 Qwen2.5-VL 视觉预处理基本一致的 smart_resize。

    核心约束：
    1. 输出高宽必须是 factor 的整数倍；
    2. 输出总像素数落在 [min_pixels, max_pixels]；
    3. 尽量保持原始长宽比。

    与官方实现的差异：
    - 当输入图像任意一边小于 factor 时，直接丢弃；
    - 官方实现不会直接丢弃，而是尽量将其缩放到合法尺寸。
    - 这里 factor=28，因为 Qwen2.5-VL 的 patch_size=14，merge_size=2，所以最终尺寸需对齐到 28 的倍数。
    """
    #太小的图片也丢弃
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    #长宽比太极端的图片丢弃
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def scale_box_xyxy(
    box: Box,
    scale_w: float,
    scale_h: float,
    new_w: int,
    new_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    x1n = int(round(x1 * scale_w))
    y1n = int(round(y1 * scale_h))
    x2n = int(round(x2 * scale_w))
    y2n = int(round(y2 * scale_h))

    #把坐标限制在区间 0到新坐标内。
    x1n = _clamp_int(x1n, 0, new_w)
    y1n = _clamp_int(y1n, 0, new_h)
    x2n = _clamp_int(x2n, 0, new_w)
    y2n = _clamp_int(y2n, 0, new_h)

    if x1n > x2n:
        x1n, x2n = x2n, x1n
    if y1n > y2n:
        y1n, y2n = y2n, y1n

    return x1n, y1n, x2n, y2n



# 缩放坐标 -> 原图坐标
def inverse_scale_box_xyxy(
    box: Box,
    scale_w: float,
    scale_h: float,
    orig_w: int,
    orig_h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    x1o = int(round(x1 / scale_w))
    y1o = int(round(y1 / scale_h))
    x2o = int(round(x2 / scale_w))
    y2o = int(round(y2 / scale_h))

    x1o = _clamp_int(x1o, 0, orig_w)
    y1o = _clamp_int(y1o, 0, orig_h)
    x2o = _clamp_int(x2o, 0, orig_w)
    y2o = _clamp_int(y2o, 0, orig_h)

    if x1o > x2o:
        x1o, x2o = x2o, x1o
    if y1o > y2o:
        y1o, y2o = y2o, y1o

    return x1o, y1o, x2o, y2o

def rescale_assistant_text_bboxes(
    assistant_text: str,
    orig_w: int,
    orig_h: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28,
    *,
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
) -> Optional[str]:
    if assistant_text is None:
        return assistant_text

    full = assistant_text.strip()
    ans = extract_answer_text(full)

    if not ans or ans.lower() == "none":
        return assistant_text

    try:
        new_h, new_w = smart_resize(orig_h, orig_w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    except ValueError:
        print(f"[WARN] Skipping sample due to small image size: h={orig_h}, w={orig_w}, factor={factor}")
        return None
    #缩放或者放大了多少
    scale_w = new_w / float(orig_w)
    scale_h = new_h / float(orig_h)

    triples = parse_triples(ans, strict=False, where="bbox_rescale")
    if not triples:
        if _structured_answer_parse_failed(ans): #格式错误就跳过 否则保留ans(ans本身为none) 当然 这两种情况都无需缩放
            print(f"[WARN] bbox_rescale parse failed -> skip sample. answer={_short_text_for_log(ans)!r}")
            return None
        return assistant_text

    new_triples: List[EntityTriple] = []
    for t in triples:
        if not t.region_valid: #region_valid为false时 表示bbox的格式出问题 直接return
            print(f"[WARN] bbox_rescale found invalid bbox -> skip sample. answer={_short_text_for_log(ans)!r}")
            return None
        if t.regions is None:
            new_triples.append(t)
            continue

        scaled_regions: List[Box] = []
        for box in t.regions:
            x1n, y1n, x2n, y2n = scale_box_xyxy(box, scale_w, scale_h, new_w, new_h)
            gt_scaled: Box = (float(x1n), float(y1n), float(x2n), float(y2n))

            #扰动
            if do_jitter:
                w = max(0.0, gt_scaled[2] - gt_scaled[0])
                h = max(0.0, gt_scaled[3] - gt_scaled[1])
                area = w * h

                beta = float(jitter_beta)
                gamma = float(jitter_gamma)
                if area < float(jitter_min_area):
                    beta *= float(jitter_small_box_scale)
                    gamma *= float(jitter_small_box_scale)

                gt_scaled = jitter_box_with_iou_guard(
                    gt_scaled,
                    W=new_w,
                    H=new_h,
                    beta=beta,
                    gamma=gamma,
                    iou_min=float(jitter_iou_min),
                    tries=int(jitter_tries),
                    min_size=2.0,
                    jitter_dist=str(jitter_dist),
                    gauss_trunc_k=float(jitter_gauss_trunc_k),
                    scale_min=float(jitter_scale_min),
                    scale_max=float(jitter_scale_max),
                )

            scaled_regions.append(gt_scaled)

        new_triples.append(EntityTriple(text=t.text, etype=t.etype, regions=scaled_regions, region_valid=True))

    new_ans = triples_to_canon_text(new_triples)
    return replace_answer_text(full, new_ans)



# 把预测/文本中的 bbox 从缩放坐标还原回原图坐标

def restore_assistant_text_bboxes_to_original(
    assistant_text: str,
    orig_w: int,
    orig_h: int,
    min_pixels: int,
    max_pixels: int,
    factor: int = 28,
) -> Optional[str]:
    if assistant_text is None:
        return assistant_text

    full = assistant_text.strip()
    ans = extract_answer_text(full)

    if not ans or ans.lower() == "none":
        return assistant_text

    try:
        new_h, new_w = smart_resize( #这里再次smart_resize为了拿到精确的缩放比例做坐标逆映射
            orig_h,
            orig_w,
            factor=factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    except ValueError:
        print(f"[WARN] restore skip due to invalid image size: h={orig_h}, w={orig_w}")
        return assistant_text

    scale_w = new_w / float(orig_w)
    scale_h = new_h / float(orig_h)

    triples = parse_triples(ans, strict=False, where="bbox_restore")
    if not triples:
        if _structured_answer_parse_failed(ans):
            print(f"[WARN] bbox_restore parse failed -> keep raw response only. answer={_short_text_for_log(ans)!r}")
            return None
        return assistant_text

    new_triples: List[EntityTriple] = []
    for t in triples:
        if not t.region_valid:
            print(f"[WARN] bbox_restore found invalid bbox -> keep raw response only. answer={_short_text_for_log(ans)!r}")
            return None
        if t.regions is None:
            new_triples.append(t)
            continue

        restored_regions: List[Box] = []
        for box in t.regions:
            x1o, y1o, x2o, y2o = inverse_scale_box_xyxy(
                box,
                scale_w=scale_w,
                scale_h=scale_h,
                orig_w=orig_w,
                orig_h=orig_h,
            )
            restored_regions.append((float(x1o), float(y1o), float(x2o), float(y2o)))

        new_triples.append(
            EntityTriple(
                text=t.text,
                etype=t.etype,
                regions=restored_regions,
                region_valid=True,
            )
        )

    new_ans = triples_to_canon_text(new_triples)
    return replace_answer_text(full, new_ans)
