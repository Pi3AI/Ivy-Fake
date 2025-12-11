"""
Description: This script is a template for evaluating AI-generated content using OpenAI's API.
Launch scirpts:
  - OPENAI_API_KEY="" OPENAI_BASE_URL="XXX" python eva_scripts.py --eva_model_name gpt-4o-mini --res_json_path ./error_item.json
  - OPENAI_API_KEY="" OPENAI_BASE_URL="XX" python eva_scripts.py --eva_model_name deepseek-chat --res_json_path ./error_item.json
"""

import time
import os
import argparse
import re

import json_repair
import json

from loguru import logger
from typing import List, Optional
from pydantic import BaseModel

import evaluate
from sentence_transformers import SentenceTransformer

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs("./logs", exist_ok=True)
logger.add("./logs/file_{time}.log")

class InferItem(BaseModel):
    rel_path: str
    raw_ground_truth: str
    label: str
    infer_result: str
    
    @classmethod
    def from_file(cls, file_path: str):
        """加载JSON文件并转换为一组数据记录对象"""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return [cls(**item) for item in raw_data]

class Tools:
    @staticmethod
    def extract_conclusion(text: str) -> str:
        first_think_pos = text.find('<conclusion>')
        last_think_pos = text.rfind('</conclusion>')
        if first_think_pos != -1 and last_think_pos != -1:
            start = first_think_pos + len('<conclusion>')
            return text[start:last_think_pos].strip()
        return text
    @staticmethod
    def extract_think_content(text):
        """提取think标签内容（带缓存）"""
        first_think_pos = text.find('<think>')
        last_think_pos = text.rfind('</think>')

        if first_think_pos != -1 and last_think_pos != -1:
            # 提取匹配到的内容
            start = first_think_pos + len('<think>')
            end = last_think_pos
            return text[start:end].strip()
        elif first_think_pos != -1 and last_think_pos == -1 and text.rfind('<conclusion>') !=-1:
          start = first_think_pos + len('<think>')
          end = text.rfind('<conclusion>')
          return text[start:end].strip()
        return ''


class Eva:
    @staticmethod
    def extract_json(response_text):
        """
        只提取 ```json ... ``` 代码块；如果没有就整体尝试 json.loads。
        """
        # 提取 markdown 中的 ```json``` 块
        match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json_repair.loads(json_str)
            except json.JSONDecodeError:
                return None  # JSON代码块存在但格式错误
        else:
            try:
                return json_repair.loads(response_text)
            except json.JSONDecodeError:
                return None  # 整体解析也失败
    @staticmethod
    def call_openai_api(model_name, prompt, system_prompt):
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content

    @staticmethod
    def eva_cls(json_item):
        
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        rouge_metric = evaluate.load("rouge")
        tos = {
            "acc": 0.0,
            "f1": 0.0,
            "rougel": 0.0,
            "sim": 0.0
        }
        y_true,y_pred = [],[]
        plau_predict,plau_gold = [],[]
        
        for item in json_item:
            label = item.label
            infer_result = item.infer_result
            predict = Tools.extract_conclusion(infer_result).strip()
        
            if f"is {label}" in predict or predict.startswith(label):
                predict = label
        
            if label == "real":
                y_true.append(1)
            elif label == "fake":
                y_true.append(2)
            else:
                y_true.append(0)
                
            if predict == "real":
                y_pred.append(1)
            elif predict == "fake":
                y_pred.append(2)
            else:
                y_pred.append(0)
                
                
            think_predict = Tools.extract_think_content(infer_result)
            gold_think = Tools.extract_think_content(item.raw_ground_truth)
            
            plau_predict.append(think_predict)
            plau_gold.append(gold_think)
            
        tos["acc"] = accuracy_metric.compute(
            references=y_true,
            predictions=y_pred
        )["accuracy"]
        
        f1 = f1_metric.compute(
            references=y_true,
            predictions=y_pred,
            # average="binary", pos_label=0
            average="macro",
            labels=[1,2]
        )["f1"]
        
        rouge = rouge_metric.compute(
            predictions=plau_gold,
            references=plau_predict,
            use_aggregator=True
        )
        rougel = rouge["rougeL"]
        
        all_sim = []
        for s1, s2 in tqdm(zip(plau_predict, plau_gold), total=len(plau_predict), desc="Computing similarities"):
            embeddings = sentence_model.encode([s1, s2])
            similarities = sentence_model.similarity(embeddings, embeddings)[0][1]
            all_sim.append(similarities)
        avg_sim = sum(all_sim)/len(all_sim)
        
        return {
            "acc": f"{tos['acc']:.4f}",
            "f1": f"{f1:.4f}",
            "rougel": f"{rougel:.4f}",
            "sim": f"{avg_sim:.4f}"
        }
    
    @staticmethod
    def eva_plau(raw_ground_truth, label, infer_result, model_name):
        prompt, system_prompt = Eva.build_combined_prompt_and_system(raw_ground_truth, infer_result)
        response_text = Eva.call_openai_api(model_name, prompt, system_prompt)
        logger.info(f"The Response is {response_text}")
        infer_conclusion = Tools.extract_conclusion(infer_result).strip()
        res_idict = Eva.extract_json(response_text)
        if label != infer_conclusion and "is " + label not in infer_conclusion:
            for key in res_idict:
                res_idict[key] = 0.0
                
        return res_idict
    
    @staticmethod
    def eva_pipeline(json_item: List[InferItem], model_name: str):
        to_res = {
            "acc": 0.0,
            "f1": 0.0,
            "rougel": 0.0,
            "sim": 0.0,
            "Completeness": 0.0,
            "Relevance": 0.0,
            "Level of Detail": 0.0,
            "Explanation": 0.0
        }
        for item in json_item:
            raw_ground_truth = item.raw_ground_truth
            label = item.label
            plau_scores = Eva.eva_plau(raw_ground_truth, label, item.infer_result, model_name)
            for plau_i in plau_scores:
                if plau_i in to_res:
                    to_res[plau_i] += plau_scores[plau_i]
        for k in to_res:
            to_res[k] = to_res[k] / len(json_item)
            
        cls_res = Eva.eva_cls(json_item)
        to_res.update(cls_res)
        return to_res

    @staticmethod
    def build_combined_prompt_and_system(ground_truth: str, model_output: str):
        prompt = f'{{"GroundTruth": "{ground_truth}", "ModelOutput": "{model_output}"}}'


        system_prompt = (
            "You are an impartial evaluator. Your task is to assess whether a model-generated response accurately and coherently matches a human-annotated reference answer.\n\n"
            "Each input contains two structured components:\n"
            "- <think>: the reasoning or analytical explanation\n"
            "- <conclusion>: the final judgment (e.g., real or fake)\n\n"
            "Compare the ModelOutput to the GroundTruth and assign integer scores from 1 to 5 (no decimals) for each of the following four evaluation dimensions:\n\n"
            "1. Completeness:\n"
            "- To what extent does the ModelOutput address all aspects covered in the GroundTruth?\n"
            "- More complete responses should incorporate information aligning well with the 'golden clues' or reference answer.\n"
            "- Incomplete or partially addressed responses should receive lower scores.\n\n"
            "2. Relevance:\n"
            "- Evaluate whether the ModelOutput discusses the same detection dimensions (e.g., temporal and spatial features) as those in the GroundTruth.\n"
            "- Temporal features include: luminance discrepancy, duplicated components, awkward facial expressions, and motion inconsistency.\n"
            "- Spatial features include: abnormal texture, distorted or omitted components, chromatic irregularity, impractical luminosity, localized blur, etc.\n"
            "- Penalize if the ModelOutput fails to address relevant dimensions from the GroundTruth, or introduces unrelated ones.\n\n"
            "3. Level of Detail:\n"
            "- Evaluate whether the ModelOutput describes the fine-grained visual cues or specific elements observed in each detection dimension (e.g., texture irregularity, glow inconsistency, blurred edges).\n"
            "- A high score requires detailed elaboration of subcomponents, not just mentioning broad categories.\n"
            "- Penalize vague, generic, or oversimplified descriptions that lack specific observations.\n\n"
            "4. Explanation:\n"
            "- Assess whether the reasoning in the <think> section logically supports the <conclusion>.\n"
            "- The explanation should provide clear and causally linked justification that leads to the final decision.\n"
            "- Penalize if the <conclusion> contradicts the reasoning or is not supported by the analysis.\n\n"
            "Scoring Instructions:\n"
            "- Use only integers from 1 to 5 for each category.\n"
            "- Do not provide any extra commentary. Output only the following JSON format:\n"
            '```json\n{\n'
            '  "Completeness": <int>,\n'
            '  "Relevance": <int>,\n'
            '  "Level of Detail": <int>,\n'
            '  "Explanation": <int>\n'
            '}\n```\n\n'
        )

        return system_prompt, prompt

def init_args():
    parser = argparse.ArgumentParser(description='Test for argparse', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--eva_model_name', '-emn', help='The modelname for evaluation', type=str, default='gpt-4o-mini')
    parser.add_argument("--res_json_path", "-rj", help="The path to the result json file", type=str)
    args = parser.parse_args()
    return args

def main(args):
    json_item = InferItem.from_file(args.res_json_path)
    
    logger.info(f"Loaded {len(json_item)} items from {args.res_json_path}")
    
    eva_ress = Eva.eva_pipeline(
        json_item=json_item,
        model_name=args.eva_model_name
    )
    logger.info(eva_ress)

if __name__ == "__main__":
    args = init_args()
  
    main(args)
