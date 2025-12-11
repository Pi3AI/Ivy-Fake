import os
import json
import re
import gc
import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from loguru import logger
from tqdm import tqdm
import argparse

# =============================
# 命令行参数
# =============================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate text similarity and metrics")
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--bert_model", type=str, required=True, help="BERTScore 模型路径")
    return parser.parse_args()

args = parse_args()

input_file = args.input_file
batch_size = args.batch_size
bert_model = args.bert_model

# =============================
# 环境变量
# =============================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['HF_HOME'] = './cache'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# =============================
# 初始化容器
# =============================
file_types = ["image", "video"]
metrics = ["Completeness", "Relevance", "Level of Detail", "Explanation"]

data = {
    ft: {"scores": defaultdict(list), "labels": [], "preds": [],
         "rougeL": [], "bertscore": []}
    for ft in file_types + ["overall"]
}

# =============================
# 工具函数
# =============================
def detect_file_type(file_path):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".mpeg", ".mpg"}
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in image_exts:
        return "image"
    elif ext in video_exts:
        return "video"
    return None

def extract_think(text):
    matchs = re.search(r"<think>(.*)</think>", text, re.IGNORECASE | re.DOTALL)
    return matchs.group(1).strip() if matchs else text

def extract_conclusion(text):
    matchs = re.search(r"<conclusion>(.*)</conclusion>", text, re.IGNORECASE | re.DOTALL)
    return matchs.group(1).strip() if matchs else text

# =============================
# Step 1: 读取数据
# =============================
print("正在统计文件总行数...")
with open(input_file, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
print(f"文件总行数: {total_lines}")

all_preds, all_refs, all_filetypes = [], [], []
labels, preds = {"overall": [], "image": [], "video": []}, {"overall": [], "image": [], "video": []}

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="读取文件", unit="行"):
        item = json.loads(line.strip())
        ft = detect_file_type(item["file_path"]) or "overall"

        response = extract_think(item["raw_ground_truth"])
        pred = extract_think(item["infer_result"])
        predict_label = extract_conclusion(item["infer_result"])
        label = item["label"]

        # 收集维度指标
        if "evaluation" in item:
            for k in metrics:
                if k in item["evaluation"]:
                    if ft in data:
                        data[ft]["scores"][k].append(item["evaluation"][k])
                    data["overall"]["scores"][k].append(item["evaluation"][k])

        if ft in data:
            data[ft]["labels"].append(label)
            data[ft]["preds"].append(predict_label)
        data["overall"]["labels"].append(label)
        data["overall"]["preds"].append(predict_label)

        all_refs.append(response)
        all_preds.append(pred)
        all_filetypes.append(ft)

# =============================
# Step 2: 批量计算 ROUGE & BERTScore
# =============================
rouge_scorer_fn = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
all_rouges, all_berts = [], []

for i in tqdm(range(0, len(all_preds), batch_size), desc="批量计算文本相似度", unit="batch"):
    batch_preds = all_preds[i:i+batch_size]
    batch_refs = all_refs[i:i+batch_size]

    # ROUGE-L
    batch_rouges = [
        rouge_scorer_fn.score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(batch_refs, batch_preds)
    ]
    all_rouges.extend(batch_rouges)

    # BERTScore
    try:
        P, R, F1 = bert_score(
            batch_preds, batch_refs,
            num_layers=24,
            model_type=bert_model,
            lang="en",
            rescale_with_baseline=False,
            batch_size=batch_size
        )
        all_berts.extend(F1.tolist())
    except Exception as e:
        logger.exception(f"计算 bert 报错，退回单条计算: {e}")
        for pred, ref in zip(batch_preds, batch_refs):
            try:
                _, _, F1 = bert_score(
                    [pred], [ref],
                    num_layers=24,
                    model_type=bert_model,
                    lang="en",
                    rescale_with_baseline=False,
                    batch_size=1
                )
                all_berts.append(float(F1.mean()))
            except Exception as e2:
                logger.warning(f"单条计算 BERT 报错，跳过：{e2}")
                continue

    # 清理缓存
    del batch_preds, batch_refs, batch_rouges
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# =============================
# Step 3: 汇总
# =============================
for ft, rouge_val, bert_val in zip(all_filetypes, all_rouges, all_berts):
    if ft in data:
        data[ft]["rougeL"].append(rouge_val)
        data[ft]["bertscore"].append(bert_val)
    data["overall"]["rougeL"].append(rouge_val)
    data["overall"]["bertscore"].append(bert_val)

# =============================
# Step 4: 输出结果
# =============================
def print_results(title, d):
    print(f"\n==== {title} ====")
    avg_scores = {k: (sum(v)/len(v) if v else 0.0) for k, v in d["scores"].items()}
    for k, v in avg_scores.items():
        print(f"{k}: {v:.2f}")

    if d["labels"] and d["preds"]:
        acc = accuracy_score(d["labels"], d["preds"])
        _, _, f1, _ = precision_recall_fscore_support(d["labels"], d["preds"], average="weighted", zero_division=0)
        print(f"\nAccuracy: {acc:.4f}")
        print(f"F1:       {f1:.4f}")
    else:
        acc, f1 = 0, 0

    rougel = sum(d["rougeL"])/len(d["rougeL"]) if d["rougeL"] else 0
    bert = sum(d["bertscore"])/len(d["bertscore"]) if d["bertscore"] else 0
    print(f"ROUGE-L:  {rougel:.4f}")
    print(f"BERTScore:{bert:.4f}")

    return (
        " & {acc:.3f}/{f1:.3f}/{rougel:.3f}/{bert:.3f}"
        " & {com:.2f}/{rel:.2f}/{det:.2f}/{exp:.2f}"
    ).format(
        acc=acc, f1=f1, rougel=rougel, bert=bert,
        com=avg_scores.get("Completeness", 0),
        rel=avg_scores.get("Relevance", 0),
        det=avg_scores.get("Level of Detail", 0),
        exp=avg_scores.get("Explanation", 0)
    )

to_res = f"{os.path.basename(input_file)}"
for ft in ["image", "video", "overall"]:
    to_res += print_results(ft.capitalize(), data[ft])

logger.info(to_res)
logger.info(f"✅ 完成计算: {input_file}")