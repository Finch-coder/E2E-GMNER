#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Convert grounded MNER raw data into SFT JSONL.

This version is cleaned up for open-source release:
- no hard-coded personal absolute paths
- config-driven via JSON file
- optional CoT output, disabled by default for safer release
- image paths in output can be relative instead of leaking local machine paths
- clearer errors and logging
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

INSTRUCTION = (
    "Perform grounded MNER by extracting entities from the text.\n"
    "For each entity, output `entity|type|[x1,y1,x2,y2]` if it is visible in the image; "
    "otherwise output `entity|type|None`.\n"
    "Separate multiple results with `;`, or output `None` if no entity is found.\n\n"
    "Use both textual evidence and visual evidence when available.\n\n"
    "Text: "
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
DEFAULT_SPLITS = ("train", "dev", "test")


class ConfigError(ValueError):
    """Raised when the config is invalid."""


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_path(p: str | Path | None) -> str:
    if not p:
        return ""
    return os.path.normpath(str(p))


def basename_no_ext(p: str | Path | None) -> str:
    if not p:
        return ""
    return Path(p).stem


def validate_config(cfg: dict[str, Any]) -> None:
    required_top = ["raw", "sft"]
    for key in required_top:
        if key not in cfg:
            raise ConfigError(f"Missing top-level config key: {key}")

    raw = cfg["raw"]
    sft = cfg["sft"]
    cot = cfg.get("cot", {})

    for key in ["xml_root", "img_roots", "splits"]:
        if key not in raw:
            raise ConfigError(f"Missing raw.{key} in config")

    if not isinstance(raw["img_roots"], list) or not raw["img_roots"]:
        raise ConfigError("raw.img_roots must be a non-empty list")

    if not isinstance(raw["splits"], dict) or not raw["splits"]:
        raise ConfigError("raw.splits must be a non-empty dict")

    if "output_dir" not in sft:
        raise ConfigError("Missing sft.output_dir in config")

    if "splits" not in sft or not isinstance(sft["splits"], dict):
        raise ConfigError("Missing sft.splits in config")

    if "data_root" in sft and not isinstance(sft["data_root"], str):
        raise ConfigError("sft.data_root must be a string if provided")

    if cot and not isinstance(cot, dict):
        raise ConfigError("cot must be an object if provided")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON parse failed: {path}:{line_no} -> {e}", file=sys.stderr)
    return records


def read_conll_file(path: str | Path) -> list[dict[str, Any]]:
    """Read GMNER-style txt data.

    Each sample starts with IMGID:xxxxx, followed by token-tag rows,
    and samples are separated by blank lines.
    """
    samples: list[dict[str, Any]] = []
    img_id: str | None = None
    tokens: list[str] = []
    tags: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                if img_id is not None and tokens:
                    samples.append({"img_id": img_id, "tokens": tokens, "tags": tags})
                    tokens, tags = [], []
                continue

            if line.startswith("IMGID:"):
                if img_id is not None and tokens:
                    samples.append({"img_id": img_id, "tokens": tokens, "tags": tags})
                    tokens, tags = [], []
                img_id = line.split(":", 1)[1].strip()
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] malformed line skipped in {path}: {raw_line.rstrip()}", file=sys.stderr)
                continue

            tokens.append(parts[0])
            tags.append(parts[-1])

    if img_id is not None and tokens:
        samples.append({"img_id": img_id, "tokens": tokens, "tags": tags})

    return samples


def tokens_to_text(tokens: list[str]) -> str:
    return html.unescape(" ".join(tokens))


def bio_to_entities(tokens: list[str], tags: list[str]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    cur_tokens: list[str] = []
    cur_type: str | None = None

    def flush() -> None:
        nonlocal cur_tokens, cur_type
        if cur_tokens and cur_type:
            entities.append({"text": " ".join(cur_tokens), "type": cur_type})
        cur_tokens = []
        cur_type = None

    for tok, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            flush()
            cur_type = tag[2:]
            cur_tokens = [tok]
        elif tag.startswith("I-"):
            typ = tag[2:]
            if cur_type == typ:
                cur_tokens.append(tok)
            else:
                flush()
                cur_type = typ
                cur_tokens = [tok]
        else:
            flush()

    flush()
    return entities


def find_image_path(img_id: str, roots: list[str | Path]) -> str | None:
    for root in roots:
        root_path = Path(root)
        for ext in IMAGE_EXTENSIONS:
            candidate = root_path / f"{img_id}{ext}"
            if candidate.exists():
                return str(candidate.resolve())
    return None
#find_image_path() 找不到图像时 img_path=None，仍会生成样本 面向该数据集可用


def xml_path_from_image(img_path: str | None, xml_root_dir: str | Path) -> str | None:
    if img_path is None:
        return None
    xml_path = Path(xml_root_dir) / f"{Path(img_path).stem}.xml"
    return str(xml_path.resolve()) if xml_path.exists() else None


def parse_voc_style_xml(xml_path: str | Path) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] failed to parse XML: {xml_path}, error: {e}", file=sys.stderr)
        return objects

    for obj in root.findall(".//object"):
        name_el = obj.find("name")
        bnd = obj.find("bndbox")
        if name_el is None or bnd is None:
            continue

        name = (name_el.text or "").strip()
        try:
            xmin = int(float((bnd.find("xmin").text)))
            ymin = int(float((bnd.find("ymin").text)))
            xmax = int(float((bnd.find("xmax").text)))
            ymax = int(float((bnd.find("ymax").text)))
        except Exception:  # noqa: BLE001
            continue

        objects.append(
            {
                "name": name,
                "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            }
        )

    return objects


def normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def match_entities_with_objects(entities, objects):
    """Match entities to XML objects by exact name equality.

    One entity can match multiple objects (multi-box).
    """
    grounded = []
    for ent in entities:
        ent_text = (ent["text"] or "").strip()
        if not ent_text:
            continue
        for obj in objects:
            obj_name = (obj["name"] or "").strip()
            if not obj_name:
                continue
            if ent_text == obj_name:
                grounded.append(
                    {
                        "text": ent["text"],
                        "type": ent["type"],
                        "bbox": obj["bbox"],
                        "object_name": obj["name"],
                    }
                )
    return grounded


def build_record_from_sample(
    sample: dict[str, Any],
    xml_root: str | Path,
    img_roots: list[str | Path],
) -> dict[str, Any]:
    img_id = sample["img_id"]
    tokens = sample["tokens"]
    tags = sample["tags"]

    text = tokens_to_text(tokens)
    entities = bio_to_entities(tokens, tags)

    img_path = find_image_path(img_id, img_roots)
    xml_path = xml_path_from_image(img_path, xml_root)

    grounded_entities: list[dict[str, Any]] = []
    if xml_path and entities:
        objects = parse_voc_style_xml(xml_path)
        grounded_entities = match_entities_with_objects(entities, objects)

    return {
        "img_id": img_id,
        "img_path": img_path,
        "xml_path": xml_path,
        "text": text,
        "entities": entities,
        "grounded_entities": grounded_entities,
        "has_entity": bool(entities),
        "has_grounding": bool(grounded_entities),
    }


def format_bbox(bbox: dict[str, int] | None) -> str:
    if not bbox:
        return "None"
    return f"[{bbox['xmin']},{bbox['ymin']},{bbox['xmax']},{bbox['ymax']}]"


def format_bboxes(bboxes: list[dict[str, int]] | None) -> str:
    if not bboxes:
        return "None"
    if len(bboxes) == 1:
        return format_bbox(bboxes[0])
    parts = []
    for b in bboxes:
        parts.append(f"[{b['xmin']},{b['ymin']},{b['xmax']},{b['ymax']}]")
    return "[" + ",".join(parts) + "]"


def build_answer_from_record(rec: dict[str, Any]) -> str:
    entities = rec.get("entities", []) or []
    grounded_entities = rec.get("grounded_entities", []) or []

    if not entities:
        return "None"

    grounded_map: dict[tuple[str, str], list[dict[str, int]]] = defaultdict(list)
    for grounded in grounded_entities:
        key = ((grounded.get("text") or "").strip(), (grounded.get("type") or "").strip())
        bbox = grounded.get("bbox")
        if bbox:
            grounded_map[key].append(bbox)

    triples: list[str] = []
    for ent in entities:
        text = (ent.get("text") or "").strip()
        ent_type = (ent.get("type") or "").strip()
        if not text or not ent_type:
            continue
        key = (text, ent_type)
        bbox_str = format_bboxes(grounded_map.get(key))
        triples.append(f"{text}|{ent_type}|{bbox_str}")

    return "; ".join(triples) if triples else "None"


def build_cot_index(cot_records: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    by_full_path: dict[str, str] = {}
    by_basename: dict[str, str] = {}

    for record in cot_records:
        image_path = normalize_path(record.get("image_path", ""))
        think = (record.get("think") or record.get("cot") or record.get("reasoning") or "").strip()
        if not think:
            continue
        if image_path:
            by_full_path[image_path] = think
        base = basename_no_ext(image_path)
        if base:
            by_basename[base] = think

    return by_full_path, by_basename


def find_think_for_record(
    rec: dict[str, Any],
    cot_by_full_path: dict[str, str],
    cot_by_basename: dict[str, str],
) -> str:
    img_path = normalize_path(rec.get("img_path", ""))
    if img_path in cot_by_full_path:
        return cot_by_full_path[img_path]

    base = basename_no_ext(img_path)
    return cot_by_basename.get(base, "")


def sanitize_image_reference(
    img_path: str,
    image_mode: str,
    image_root: str | Path | None = None,
) -> str:
    if image_mode == "filename":
        return Path(img_path).name
    if image_mode == "relative":
        if image_root is None:
            raise ValueError("image_root is required when image_mode='relative'")
        return os.path.relpath(img_path, str(Path(image_root).resolve()))
    if image_mode == "absolute":
        return img_path
    raise ValueError(f"Unsupported image_mode: {image_mode}")


def convert_record(
    rec: dict[str, Any],
    sample_id: int,
    include_cot: bool,
    image_mode: str,
    image_root: str | Path | None,
    cot_by_full_path: dict[str, str] | None = None,
    cot_by_basename: dict[str, str] | None = None,
) -> dict[str, Any]:
    text = (rec.get("text") or "").strip()
    img_path = (rec.get("img_path") or "").strip()
    answer = build_answer_from_record(rec)
    user_content = INSTRUCTION + text

    if include_cot:
        think = find_think_for_record(rec, cot_by_full_path or {}, cot_by_basename or {})
        assistant_content = f"<think>{think}</think><answer>{answer}</answer>"
    else:
        assistant_content = answer

    output: dict[str, Any] = {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }

    if img_path:
        output["images"] = [
            sanitize_image_reference(
                img_path,
                image_mode=image_mode,
                image_root=image_root,
            )
        ]
    else:
        output["images"] = []

    return output


def init_stats() -> dict[str, Any]:
    return {
        "total_samples": 0,
        "samples_with_entities": 0,
        "samples_without_entities": 0,
        "samples_with_grounding": 0,
        "samples_without_grounding": 0,
        "total_entities": 0,
        "entities_with_bbox": 0,
        "entities_without_bbox": 0,
        "total_boxes": 0,
        "type_counts": Counter(),
    }


def update_stats(rec: dict[str, Any], stats: dict[str, Any]) -> None:
    entities = rec.get("entities", []) or []
    grounded_entities = rec.get("grounded_entities", []) or []

    stats["total_samples"] += 1
    stats["samples_with_entities"] += int(bool(entities))
    stats["samples_without_entities"] += int(not entities)
    stats["samples_with_grounding"] += int(bool(grounded_entities))
    stats["samples_without_grounding"] += int(not grounded_entities)
    stats["total_boxes"] += len(grounded_entities)

    grounded_count_map: dict[tuple[str, str], int] = defaultdict(int)
    for grounded in grounded_entities:
        key = ((grounded.get("text") or "").strip(), (grounded.get("type") or "").strip())
        grounded_count_map[key] += 1

    for ent in entities:
        text = (ent.get("text") or "").strip()
        ent_type = (ent.get("type") or "").strip()
        if not text or not ent_type:
            continue

        stats["total_entities"] += 1
        stats["type_counts"][ent_type] += 1

        key = (text, ent_type)
        if grounded_count_map[key] > 0:
            stats["entities_with_bbox"] += 1
            grounded_count_map[key] -= 1
        else:
            stats["entities_without_bbox"] += 1


def print_stats(stats: dict[str, Any], split_name: str) -> None:
    print(f"\n===== {split_name} Stats =====")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Samples with entities: {stats['samples_with_entities']}")
    print(f"Samples without entities: {stats['samples_without_entities']}")
    print(f"Samples with grounding: {stats['samples_with_grounding']}")
    print(f"Samples without grounding: {stats['samples_without_grounding']}")
    print(f"Total entities: {stats['total_entities']}")
    print(f"Groundable Entity: {stats['entities_with_bbox']}")
    print(f"Entities without bbox: {stats['entities_without_bbox']}")
    print(f"Total boxes: {stats['total_boxes']}")
    print("Type counts:", dict(stats["type_counts"]))


def process_split(
    split_name: str,
    txt_path: str | Path,
    cot_path: str | Path | None,
    out_path: str | Path,
    xml_root: str | Path,
    img_roots: list[str | Path],
    start_id: int,
    include_cot: bool,
    image_mode: str,
    image_root: str | Path | None,
) -> int:
    txt_path = str(txt_path)
    out_path = str(out_path)

    if not os.path.exists(txt_path):
        print(f"[WARN] raw txt not found, skip {split_name}: {txt_path}", file=sys.stderr)
        return start_id

    samples = read_conll_file(txt_path)
    print(f"[{split_name}] loaded raw samples: {len(samples)}")

    cot_by_full_path: dict[str, str] = {}
    cot_by_basename: dict[str, str] = {}
    missing_cot = 0
    if include_cot and cot_path:
        if os.path.exists(cot_path):
            cot_by_full_path, cot_by_basename = build_cot_index(load_jsonl(cot_path))
        else:
            print(f"[WARN] CoT file not found for {split_name}: {cot_path}", file=sys.stderr)

    stats = init_stats()
    cur_id = start_id

    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples, 1):
            rec = build_record_from_sample(sample, xml_root=xml_root, img_roots=img_roots)
            update_stats(rec, stats)

            if include_cot and not find_think_for_record(rec, cot_by_full_path, cot_by_basename):
                missing_cot += 1

            converted = convert_record(
                rec=rec,
                sample_id=cur_id,
                include_cot=include_cot,
                image_mode=image_mode,
                image_root=image_root,
                cot_by_full_path=cot_by_full_path,
                cot_by_basename=cot_by_basename,
            )
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            cur_id += 1

            if idx % 100 == 0:
                print(f"[{split_name}] processed {idx}/{len(samples)}")

    print_stats(stats, split_name)
    print(f"[{split_name}] output: {out_path}")
    print(
        f"[{split_name}] samples without matched CoT: {missing_cot}"
        if include_cot
        else f"[{split_name}] CoT disabled"
    )
    return cur_id


def run(cfg: dict[str, Any], include_cot: bool, image_mode: str) -> None:
    validate_config(cfg)

    raw_cfg = cfg["raw"]
    cot_cfg = cfg.get("cot", {})
    sft_cfg = cfg["sft"]

    output_dir = Path(sft_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_root = raw_cfg["xml_root"]
    img_roots = raw_cfg["img_roots"]

    # relative 模式统一相对于 data_root
    # 没显式配置时，默认用 output_dir.parent
    data_root = Path(sft_cfg.get("data_root", output_dir.parent)).resolve()
    image_root_for_relative = data_root if image_mode == "relative" else None

    global_id = 0
    for split_name in DEFAULT_SPLITS:
        if split_name not in raw_cfg["splits"] or split_name not in sft_cfg["splits"]:
            continue

        global_id = process_split(
            split_name=split_name,
            txt_path=raw_cfg["splits"][split_name],
            cot_path=cot_cfg.get(split_name),
            out_path=output_dir / sft_cfg["splits"][split_name],
            xml_root=xml_root,
            img_roots=img_roots,
            start_id=global_id,
            include_cot=include_cot and split_name != "test",
            image_mode=image_mode,
            image_root=image_root_for_relative,
        )

    print(f"\nDone. Total written samples: {global_id}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert grounded MNER data to SFT JSONL.")
    parser.add_argument("--config", required=True, help="Path to JSON config file.")
    parser.add_argument(
        "--include-cot",
        action="store_true",
        help="Include <think>...</think> in assistant outputs. Disabled by default for open-source release.",
    )
    parser.add_argument(
        "--image-mode",
        choices=["filename", "relative", "absolute"],
        default="relative",
        help="How image paths are stored in output JSONL. Default: relative.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    cfg = load_json(args.config)
    run(cfg, include_cot=args.include_cot, image_mode=args.image_mode)


if __name__ == "__main__":
    main()
