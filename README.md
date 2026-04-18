# E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition

<p align="center">
  <img src="src/Model Architecture.jpg" alt="E2E-GMNER Framework" width="95%">
</p>


<p align="center">
  <b>Overall architecture of E2E-GMNER</b>
</p>

---

## 📌 News

- **2026-04-17**: Initial release of code and training scripts.
- **2026-04-07**: This work was accepted by **Findings of ACL 2026**.

---

## ✨ Highlights

- Reformulates **Grounded Multimodal Named Entity Recognition (GMNER)** as an **instruction-tuned conditional generation task**
- Unifies **entity recognition**, **entity typing**, and **visual grounding** in a single end-to-end framework
- Introduces **Chain-of-Thought (CoT) reasoning** for adaptive multimodal and knowledge-aware inference
- Proposes **Gaussian Risk-Aware Box Perturbation (GRBP)** for robust generative bounding box supervision

---

## 🧠 Overview

Grounded Multimodal Named Entity Recognition (GMNER) aims to jointly identify entity mentions in text, predict their semantic types, and ground each entity to the corresponding region in an associated image. Existing approaches are mostly pipeline-based, which may suffer from error propagation and suboptimal joint optimization.

To address these limitations, we propose **E2E-GMNER**, an end-to-end framework that reformulates GMNER as an **instruction-tuned conditional generation task** within a multimodal large language model. Our method further incorporates:

- **Chain-of-Thought (CoT) reasoning** to enhance adaptive multimodal and knowledge-aware reasoning
- **Gaussian Risk-Aware Box Perturbation (GRBP)** to improve the robustness of generative bounding box supervision

This unified design enables joint optimization of entity recognition, semantic typing, visual grounding, and reasoning within a single framework.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Finch-coder/E2E-GMNER.git
cd E2E-GMNER
```

### 2. Create the environment

```bash
conda create -n e2e-gmner python=3.10 -y
conda activate e2e-gmner
```


```bash
#Please install the appropriate PyTorch version for your CUDA version.
# CUDA 12.8  
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install peft
```

If you experience network issues during installation, you may use a mirror source. For example, for CUDA 12.8:
```bash
pip install torch==2.9.1 torchvision==0.24.1 --find-links https://mirrors.aliyun.com/pytorch-wheels/
```

**NOTICE**:
This project directly depends only on `torch`, `torchvision`, and `peft`.  
When installing these packages, `pip` will automatically install their required dependencies.
The `requirements.txt` file contains the full list of packages present in the environment after installing the dependencies above, and is provided for reference only.  
It is **not recommended** to install dependencies via: `pip install -r requirements.txt`

---


## 📦 Dataset

We evaluate our method on the **Twitter-GMNER** benchmark.

The dataset contains image-text pairs annotated with:

- entity spans
- semantic entity types
- grounding bounding boxes

Twitter-GMNER uses four coarse-grained entity types.

### Data Preparation

Please first download the following resources:

Please first download the required resources from the [Twitter-GMNER dataset repository](https://github.com/NUSTM/GMNER/blob/main/README.md), including **IJCAI2019_data** and **Twitter10000_v2.0**.

The CoT data is available in `data/cot/teacher_cot_train.jsonl`.

Organize the dataset in the following structure:

```text
data/
├── cot
│   └── teacher_cot_train.jsonl
└── raw
    ├── IJCAI2019_data
    │   ├── twitter2015_images
    │   └── twitter2017_images
    └── Twitter10000_v2.0
        ├── txt
        └── xml
```

Then modify `configs/data/data_config.json` according to your local paths:

```json
{
  "raw": {
    "splits": {
      "train": "data/raw/Twitter10000_v2.0/txt/train.txt",
      "dev": "data/raw/Twitter10000_v2.0/txt/dev.txt",
      "test": "data/raw/Twitter10000_v2.0/txt/test.txt"
    },
    "xml_root": "data/raw/Twitter10000_v2.0/xml",
    "img_roots": [
      "data/raw/IJCAI2019_data/twitter2015_images",
      "data/raw/IJCAI2019_data/twitter2017_images"
    ]
  },
  "cot": {
    "train": "data/cot/teacher_cot_train.jsonl",
    "dev": null,
    "test": null
  },
  "sft": {
    "output_dir": "data/sft",
    "data_root": "data",
    "splits": {
      "train": "train_sft.jsonl",
      "dev": "dev_sft.jsonl",
      "test": "test_sft.jsonl"
    }
  }
}
```

### Convert GMNER data to SFT format

Run the following command:

```bash
python scripts/gmner_to_sft_data.py \
  --config configs/data/data_config.json \
  --include-cot \
  --image-mode relative
```

The processed files will be saved as:

```text
data/
└── sft
    ├── train_sft.jsonl
    ├── dev_sft.jsonl
    └── test_sft.jsonl
```

---


## Download Pretrain_Model

Please download the pretrained backbone **Qwen2.5-VL-7B-Instruct** before training.

### Option A: Download from Hugging Face (recommended)

```bash
pip install -U "huggingface_hub[cli]"
mkdir -p model/Qwen2.5-VL-7B-Instruct
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir model/Qwen2.5-VL-7B-Instruct \
  --local-dir-use-symlinks False
```

### Option B: Download from ModelScope (for users with limited HF access)

```bash
pip install -U modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', local_dir='model/Qwen2.5-VL-7B-Instruct')"
```

After downloading, make sure `MODEL_NAME` in `scripts/train_lora.sh` points to the local model directory:
 
```bash
MODEL_NAME=model/Qwen2.5-VL-7B-Instruct
```
## 🏋️ Training

Before training, modify the following variables in `scripts/train_lora.sh`:

- `MODEL_NAME`
- `TRAIN_JSONL`
- `DEV_JSONL`
- `TEST_JSONL`
- `IMAGE_ROOT`
- `OUTPUT_DIR`

Example:

```bash
# 显存32g
#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${MODEL_NAME:-model/Qwen2.5-VL-7B-Instruct}
TRAIN_JSONL=${TRAIN_JSONL:-data/sft/train_sft.jsonl}
DEV_JSONL=${DEV_JSONL:-data/sft/dev_sft.jsonl}
TEST_JSONL=${TEST_JSONL:-data/sft/test_sft.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/exp_cot}
DEVICE=${DEVICE:-cuda:0}

python train.py \
  --model_name "$MODEL_NAME" \
  --train_jsonl "$TRAIN_JSONL" \
  --dev_jsonl "$DEV_JSONL" \
  --test_jsonl "$TEST_JSONL" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs 10 \
  --batch_size 2 \
  --grad_accum_steps 8 \
  --lr 5e-4 \
  --weight_decay 0.1 \
  --warmup_ratio 0.03 \
  --max_length 2048 \
  --bf16 \
  --grad_ckpt \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --dev_batch_size 2 \
  --test_batch_size 2 \
  --eval_max_new_tokens 512 \
  --eval_iou 0.5 \
  --num_workers 4 \
  --train_bbox_jitter \
  --jitter_beta 0.1 \
  --jitter_gamma 0.1
```

Start training with:

```bash
bash scripts/train_lora.sh
```

---



## 📊 Evaluation

Run evaluation with:

```bash
bash scripts/eval.sh
```

Before running, set (or modify in `scripts/eval.sh`) the following variables:

- `CHECKPOINT`: trained checkpoint path (e.g., `outputs/exp_cot/best`)
- `MODEL_NAME`: base model path (e.g., `model/Qwen2.5-VL-7B-Instruct`)
- `TEST_JSONL`: test split jsonl
- `IMAGE_ROOT`: image root directory
- `OUTPUT_DIR`: evaluation output directory
- `TAG`: output file suffix
- `DEVICE`: inference device (e.g., `cuda:0`)

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-outputs/exp_cot/best}
MODEL_NAME=${MODEL_NAME:-model/Qwen2.5-VL-7B-Instruct}
TEST_JSONL=${TEST_JSONL:-data/sft/test_sft.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-data}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/exp_cot/eval}
TAG=${TAG:-best_eval}
DEVICE=${DEVICE:-cuda:0}

python eval.py \
  --checkpoint "$CHECKPOINT" \
  --model_name "$MODEL_NAME" \
  --test_jsonl "$TEST_JSONL" \
  --image_root "$IMAGE_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --tag "$TAG" \
  --device "$DEVICE" \
  --test_batch_size 2 \
  --num_workers 4 \
  --max_length 2048 \
  --eval_max_new_tokens 512 \
  --eval_iou 0.5 \
  --bf16

```

Default evaluation settings in `scripts/eval.sh`:

- `--test_batch_size 2`
- `--num_workers 4`
- `--max_length 2048`
- `--eval_max_new_tokens 512`
- `--eval_iou 0.5`
- `--bf16`

Evaluation outputs are saved in `OUTPUT_DIR`:

- `test_pred_${TAG}.jsonl`: generated predictions
- `test_metrics_${TAG}.json`: aggregated metrics
- `eval_results.txt`: evaluation log

Reported metrics include:

- **GMNER** (entity + type + region): Precision / Recall / F1
- **MNER** (entity + type): Precision / Recall / F1
- **EEG** (entity + region): Precision / Recall / F1


---

## 🗂️ Project Structure

```text
E2E-GMNER/
├── configs/
├── data/
├── outputs/
├── scripts/
├── src/
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```

---

## 🙏 Acknowledgements

We sincerely thank the authors of the following open-source projects and datasets:

- [GMNER / H-Index](https://github.com/NUSTM/GMNER) for providing the dataset
- [RiVEG](https://github.com/JinYuanLi0012/RiVEG)
- [MAKAR](https://github.com/Nikol-coder/MAKAR)

Our implementation is built upon these excellent works.

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@article{e2e_gmner_2026,
  title={E2E-GMNER: End-to-End Generative Grounded Multimodal Named Entity Recognition},
  author={Anonymous},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

---

## 📮 Contact

For questions or collaborations, please open an issue or contact the authors.
