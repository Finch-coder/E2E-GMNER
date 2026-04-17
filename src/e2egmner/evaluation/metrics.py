"""Matching and metric computation for GMNER, MNER, and EEG."""

from typing import Dict, List, Tuple

from .geometry import region_correct
from .parsing import EntityTriple

def _best_bipartite_match_count_by_score(num_preds: int, num_golds: int, edges: Dict[Tuple[int, int], float]) -> int:
    """
    在所有合法匹配里，优先最大化匹配数量；若数量相同，再最大化总分。
    这里每个样本的实体数通常不大，用 bitmask DP 足够稳妥，也避免贪心次优。
    解决生成式方法的弊端
    # 例子：
    # preds = [p0, p1], golds = [g0, g1]
    # 合法边及分数：
    #   (p0,g0)=0.90, (p0,g1)=0.80, (p1,g0)=0.70
    # 注意 p1 不能连 g1（无边）。
    #
    # 若贪心先给 p0 选最高分 g0(0.90)，则 p1 只能空着 => 只匹配 1 个。
    # 但最优解是 p0->g1(0.80), p1->g0(0.70) => 匹配 2 个。
    # 本函数的目标是：
    # 1) 先最大化匹配数量（这里选 2 个）；
    # 2) 数量相同时，再比较总分谁更大。
        """
    from functools import lru_cache

    @lru_cache(None)
    def dp(i: int, used_mask: int) -> Tuple[int, float]:
        if i >= num_preds:
            return (0, 0.0)

        best = dp(i + 1, used_mask)  # skip pred i
        for j in range(num_golds):
            if ((used_mask >> j) & 1) != 0:
                continue
            score = edges.get((i, j), None)
            if score is None:
                continue
            sub_cnt, sub_score = dp(i + 1, used_mask | (1 << j))
            cand = (sub_cnt + 1, sub_score + float(score))
            if cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
        return best

    return int(dp(0, 0)[0])


def count_correct_gmner(preds: List[EntityTriple], golds: List[EntityTriple], iou_threshold: float = 0.5) -> int:
    edges: Dict[Tuple[int, int], float] = {}
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            if p.text != g.text or p.etype != g.etype:
                continue
            ok, score = region_correct(p, g, iou_threshold)
            if ok:
                edges[(i, j)] = float(score)
    return _best_bipartite_match_count_by_score(len(preds), len(golds), edges)


def count_correct_mner(preds: List[EntityTriple], golds: List[EntityTriple]) -> int:
    used = [False] * len(golds)
    correct = 0
    for p in preds:
        for j, g in enumerate(golds):
            if used[j]:
                continue
            if p.text == g.text and p.etype == g.etype:
                used[j] = True
                correct += 1
                break
    return correct


def count_correct_eeg(preds: List[EntityTriple], golds: List[EntityTriple], iou_threshold: float = 0.5) -> int:
    edges: Dict[Tuple[int, int], float] = {}
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            if p.text != g.text:
                continue
            ok, score = region_correct(p, g, iou_threshold)
            if ok:
                edges[(i, j)] = float(score)
    return _best_bipartite_match_count_by_score(len(preds), len(golds), edges)


def prf(correct: int, predicted: int, gold: int):
    p = correct / predicted if predicted > 0 else 0.0
    r = correct / gold if gold > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1
