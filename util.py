import os
import json
import random
import requests
import string

from loguru import logger
from tqdm import tqdm
from typing import List, Optional, Tuple
from transformers import AutoTokenizer
from dataclasses import dataclass, field, asdict

HF_DATASET_PRESET = {
    "sharegpt_wizard": "Thermostatic/ShareGPT_wizard_vicuna_unfiltered_no_alignment",
    "sharegpt_vicuna": "anon8231489123/ShareGPT_Vicuna_unfiltered",
    "cnn_dailymail": "cnn_dailymail:3.0.0",
    "dolly": "databricks/databricks-dolly-15k",
    "alpaca": "yahma/alpaca-cleaned",
    "alpaca_code": "iamtarun/python_code_instructions_18k_alpaca",
}

OPENROUTER_EP = "https://openrouter.ai/api/v1"

CNN_DAILYMAIL_SYS_PROMPT = "As an AI assist, could you please summarize or highlight the following content "
ALPACA_CODE_SYS_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

@dataclass
class LlmProvider:
    provider: str = field(default="")
    model: str = field(default="")
    endpoint: str = field(default="")
    api_key: Optional[str] = field(default=None)
    def is_openrouter(self):
        return self.provider.startswith("openrouter:")
    def get_openrouter_provider(self):
        if self.provider.startswith("openrouter:"):
            return self.provider[11:]
        else:
            return None

def get_hf_dataset_path(key: str) -> str:
    if key in HF_DATASET_PRESET:
        return HF_DATASET_PRESET[key]
    return None

def get_context_with_fewshot(ds, doc, rnd, num_fewshot, doc_to_text, doc_to_target):
    n_samples = num_fewshot + 1
    fewshotex = rnd.sample(ds, n_samples)
    selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]
    labeled_examples = ""
    for doc in selected_docs:
        doc_content = doc_to_text(doc)
        labeled_examples += doc_content
        doc_target = doc_to_target(doc)
        if doc_target != "":
            labeled_examples += " "
            labeled_examples += (str(doc_target[0]) if isinstance(doc_target, list) else doc_target)
            labeled_examples += "\n\n"
    return labeled_examples + doc_to_text(doc)

def load_hf_dataset(ds_path, split="train"):
    from datasets import Dataset, load_dataset, VerificationMode
    if os.path.exists(ds_path):
        with open(ds_path) as f:
            dataset = json.load(f)
    else:
        logger.info(f"Downloading dataset from huggingface: {ds_path}")
        parts = ds_path.split(":")
        if len(parts) == 1:
            dataset = load_dataset(ds_path, split=split, verification_mode=VerificationMode.NO_CHECKS)
        elif len(parts) == 2:
            data_name, subset_name = parts
            dataset = load_dataset(data_name, subset_name, split="test")
        else:
            raise RuntimeError("Dataset name is in invalid format. (valid fmt: '<dataset_name>' or '<dataset_name>:<subset_name>')")
    return dataset

def load_sharegpt_dataset(ds_path: str) -> List[Tuple[str, str]]:
    # Get dataset dir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Find local sharegpt dataset file
    local_sharegpt_file = None
    for fname in os.listdir(dataset_dir):
        if "sharegpt" in fname.lower() and fname.endswith(".json"):
            local_sharegpt_file = os.path.join(dataset_dir, fname)
            logger.info(f"Found local ShareGPT dataset: {local_sharegpt_file}")
            break
    
    if local_sharegpt_file:
        # Directly load from local file
        with open(local_sharegpt_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        # If not, download from huggingface
        hf_ds_name = "Aeala/ShareGPT_Vicuna_unfiltered"
        logger.info(f"No local ShareGPT dataset found, downloading from huggingface: {hf_ds_name}")
        from datasets import load_dataset, VerificationMode
        hf_dataset = load_dataset(hf_ds_name, split="train", verification_mode=VerificationMode.NO_CHECKS)
        dataset = [dict(item) for item in hf_dataset]
        
        # save into local file
        save_path = os.path.join(dataset_dir, "ShareGPT_V3_unfiltered_cleaned_split.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False)
        logger.info(f"ShareGPT dataset saved to: {save_path}")
    """Load ShareGPT dataset and return list of tuple(prompt, response) pairs"""
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset]
    random.shuffle(dataset)
    return dataset

def load_cnn_dailymail_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load CNN_Dailymail dataset and return list of tuple(prompt, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    dataset = [(CNN_DAILYMAIL_SYS_PROMPT + data["article"], data["highlights"]) for data in dataset]
    random.shuffle(dataset)
    return dataset

def load_dolly_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load Dolly dataset and return list of tuple(prompt, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    dataset = [(data["instruction"], data["response"]) for data in dataset]
    random.shuffle(dataset)
    return dataset

def load_alpaca_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load Alpaca dataset and return list of tuple(prompt, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    dataset = [(data["instruction"] + " " + data["input"], data["output"]) for data in dataset]
    random.shuffle(dataset)
    return dataset

def load_alpaca_code_dataset(ds_path: str) -> List[Tuple[str, str]]:
    """Load Alpaca coding dataset and return list of tuple(instruct, input, response) pairs"""
    dataset = load_hf_dataset(ds_path)
    result = []
    for data in dataset:
        prompt = ALPACA_CODE_SYS_PROMPT + " ### Instruction: " + data["instruction"] + " ### Input: " + data["input"]
        result.append((prompt, data["output"]))
    random.shuffle(result)
    return result

def load_mmlu_pro_dataset(total_items:int = 0) -> List[Tuple[str, str]]:
    """Load MMLU-Pro dataset and return list of tuple(prompt, response) pairs"""
    dataset = load_hf_dataset("TIGER-Lab/MMLU-Pro", split="test")
    items = [x for x in dataset]
    def _doc_to_text(doc):
        ret = f"{doc['question']}\n"
        for i in range(len(doc["options"])):
            ret += f"{string.ascii_uppercase[i]}. {doc['options'][i]}\n"
        ret += "Answer:"
        return ret
    def _doc_to_target(doc):
        return doc["answer"]
    rnd = random.Random()
    num_fewshot = 3
    result = []
    if total_items > 0:
        items = rnd.sample(items, total_items)
    for doc in items:
        input = get_context_with_fewshot(items, doc, rnd, num_fewshot, _doc_to_text, _doc_to_target)
        result.append((input, ""))
    return result

def load_mbpp_dataset(total_items:int=0) -> List[Tuple[str, str]]:
    """Load MBPP dataset from google-research-datasets/mbpp and return list of tuple(prompt, response) pairs"""
    ds = load_hf_dataset("google-research-datasets/mbpp", split="test")
    items = [x for x in ds]
    def _doc_to_text(doc):
        ret = f"You are an expert Python programmer, and here is your task: {doc['text']} \n Your code should pass these tests:\n\n{doc['test_list'][0]}\n{doc['test_list'][1]}\n{doc['test_list'][2]}\n[BEGIN]\n"
        return ret
    def _doc_to_target(doc):
        return ""
    rnd = random.Random()
    num_fewshot = 0
    items = [x for x in ds]
    result = []
    if total_items > 0:
        items = rnd.sample(items, total_items)
    for doc in items:
        input = get_context_with_fewshot(items, doc, rnd, num_fewshot, _doc_to_text, _doc_to_target)
        result.append((input, doc["code"]))
    return result

def filter_samples_from_dataset(dataset: List[Tuple[str, str]], tokenizer:AutoTokenizer, num_reqs:int, in_min_len:List[int], in_max_len:List[int], out_min_len:List[int], out_max_len:List[int]) -> List[Tuple[str, int, int]]:
    output = []
    if len(dataset) == 0:
        return output
    random.shuffle(dataset)
    pb = tqdm(total=num_reqs, smoothing=0.0)
    def _adjust_prompt(tokenizer, prompt, output, min_len, max_len, min_len_o, max_len_o):
        if min_len is None or min_len_o is None:
            return None, None, None
        max_len = min_len if max_len is None else max_len
        max_len_o = min_len_o if max_len_o is None else max_len_o
        prompt_tokens = tokenizer.encode(prompt)
        output_tokens = tokenizer.encode(output)
        prompt_len = len(prompt_tokens)
        output_len = len(output_tokens)
        if prompt_len < min_len:
            return None, None, None
        if prompt_len > max_len:
            prompt_len = random.randint(min_len, max_len)
            prompt = tokenizer.decode(prompt_tokens[0:prompt_len])
        if output_len > max_len_o or output_len < min_len_o:
            output_len = random.randint(min_len_o, max_len_o)
        return prompt, prompt_len, output_len

    for i in range(len(dataset)):
        if len(output) >= num_reqs:
            break
        data = dataset[i]
        n = len(output)
        prompt, prompt_len, output_len = _adjust_prompt(tokenizer, data[0], data[1], in_min_len[n], in_max_len[n], out_min_len[n], out_max_len[n])
        if prompt is not None:
            output.append((prompt, prompt_len, output_len))
            pb.update(1)
    pb.close()
    return output


def load_requests_from_json(tokenizer:AutoTokenizer, path:str, num_reqs:int, in_min_len:List[int], in_max_len:List[int], out_min_len:List[int], out_max_len:List[int]) -> List[Tuple[str, int, int]]:
    output = []
    if tokenizer is None:
        raise RuntimeError("Invalid tokenizer")
    if not os.path.exists(path):
        raise RuntimeError(f"Not exist dataset: {path}")
    if num_reqs <= 0 or num_reqs != len(in_min_len) or num_reqs != len(in_max_len) or num_reqs != len(out_min_len) or num_reqs != len(out_max_len):
        raise RuntimeError(f"Invalid argument: {num_reqs}, {len(in_min_len)}, {len(in_max_len)}, {len(out_min_len)}, {len(out_max_len)}")
    logger.info(f"Load from dataset: {path}")
    dataset = []
    with open(path, "r") as f:
        data = json.load(f)
    if "kind" in data and data["kind"] == "ppio-internal":
        for d in data["data"]:
            dataset.append((d["prompt"], d["output"]))
    elif "sharegpt" in path.lower() or "share_gpt" in path.lower():
        dataset = [d for d in data if len(d["conversations"]) >= 2]
        dataset = [(data["conversations"][0]["value"], data["conversations"][1]["value"]) for data in dataset if len(data["conversations"][0]["value"]) > 10 and len(data["conversations"][1]["value"]) > 10]
    logger.info(f"The dataset has {len(dataset)} samples")
    output = filter_samples_from_dataset(dataset, tokenizer, num_reqs, in_min_len, in_max_len, out_min_len, out_max_len)
    return output

def get_model(url: str, headers = None)->Optional[str]:
    res = requests.get(url, headers = headers)
    model_list = res.json().get("data", [])
    return model_list[0]["id"] if model_list else None

def get_model_list(url: str, headers = None)->Optional[List[str]]:
    res = requests.get(url, headers = headers)
    model_list = res.json().get("data", [])
    if model_list and len(model_list) > 0:
        models = []
        for m in model_list:
            models.append(m["id"])
        return models
    else:
        return None

def get_llm_provider(config_file:str) -> List[LlmProvider]:
    providers = []
    if os.path.isfile(config_file):
        with open(config_file, "r") as f:
            json_data = json.load(f)
            assert json_data, "Invalid provider config file"
            for p in json_data["providers"]:
                if "enable" in p and p["enable"] == False:
                    continue
                name = p["name"]
                if name.startswith("openrouter:"):
                    ep = OPENROUTER_EP
                else:
                    ep = p["endpoint"]
                for n in p["model-names"]:
                    ep2 = ep
                    if "{model}" in ep:
                        ep2 = ep.replace("{model}", n)
                    providers.append(LlmProvider(name, n, ep2, p["api_key"]))
    return providers

