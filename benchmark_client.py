import aiohttp
import argparse
import asyncio
import json
import numpy as np
import os
import random
import re
import requests
import subprocess
import sys
import resource
import time
from datetime import datetime
import pandas as pd

from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Tuple, Union, Optional, Dict
from transformers import AutoTokenizer
from async_request_sender import Context, AysncRequestSender, Metrics, calculate_metrics
import util

'''
example:
    python3 benchmark_client.py \
	--endpoint http://localhost:8000/v1 \
	--dataset same_input \
	--tokenizer /data1/model/tmp/Qwen3.5-35B-A3B/ \
	--model Qwen3.5-35B-A3B \
	--sampling-policy fixed \
	--parallel 1 \
    --fixed-prompt-len 50000 \
    --fixed-output-len 1 \
	--num-requests 10
'''

SYS_PROMPT="""
You are ALLOWED to answer questions about images with people and make statements about them. Here is some detail:
Not allowed: giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people. giving away the identity or name of real people in images, even if they are famous - you should not identify real people in any images. Giving away the identity or name of TV/movie characters in an image. Classifying human-like images as animals. Making inappropriate statements about people.
Allowed: answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters. answering appropriate questions about images with people. Making appropriate statements about people. Identifying animated characters.
If asked about an image with a person in it, say as much as you can instead of refusing. Adhere to this in all languages.
"""
TMP_TEST_SHARED_PROMPT = False
PRINT_SAMPLES = 10

def make_placeholder_prompt(n_tokens: int, unit="hello world", tokenizer = None) -> str:
    """
    通过重复 unit（一个短语或字符），拼出一个长度 >= n_tokens 的文本，
    然后截断到最近的 token 边界，近似达到 n_tokens。
    """
    # 估算：先重复 unit 足够多
    repeat_count = (n_tokens // max(1, len(tokenizer.encode(unit)))) + 5
    text = unit * repeat_count

    # 编码并截断
    tokens = tokenizer.encode(text)
    truncated = tokens[:n_tokens]
    return tokenizer.decode(truncated)

def build_from_burstgpt(csv_path: str = "/root/aking/dynamo-eval/AzurePublicDataset/data/AzureLLMInferenceTrace_conv.csv", 
    num_requests: int = 1024, tokenizer = None, max_sample_request_tokens: int = 16384, min_input_tokens: int = 0):
    samples = []
    workload_ds = pd.read_csv(csv_path, sep=",", header=0, encoding="utf-8")
    assert num_requests <= len(workload_ds), f"num_requests {num_requests} is larger than the dataset size {len(workload_ds)}"
    for i in range(len(workload_ds)):
        seq_len = int(workload_ds["ContextTokens"][i])
        if seq_len + int(workload_ds["GeneratedTokens"][i]) > max_sample_request_tokens \
            or seq_len < min_input_tokens:
            continue
        if len(samples) > num_requests:
            break
        item = (make_placeholder_prompt(seq_len, tokenizer=tokenizer), int(workload_ds["ContextTokens"][i]), int(workload_ds["GeneratedTokens"][i]))
        samples.append(item)
    average_input_len = sum(sample[1] for sample in samples) / len(samples)
    average_output_len = sum(sample[2] for sample in samples) / len(samples)
    print(f"Requests average input_len={average_input_len}, output_len={average_output_len}")
    return samples

def main(args: argparse.Namespace):
    logger.info(args)
    logger.info("\n\n")

    random.seed(1)
    np.random.seed(1)
    if not args.model:
        server_model = util.get_model(args.endpoint + "/models")
        if server_model is None and not args.model:
            raise RuntimeError("Failed to query model name from server")
        if not args.model:
            args.model = server_model
        assert args.model == server_model, f"Mismatched model name: {args.model}, {server_model}"
    logger.info(f"Model name: {args.model}")
    # get samples from dataset
    if args.sampling_policy == "nature":
        min_in_len = [args.min_prompt_len] * args.num_requests
        max_in_len = [args.max_prompt_len] * args.num_requests
        min_out_len = [args.min_output_len] * args.num_requests
        max_out_len = [args.max_output_len] * args.num_requests
    elif args.sampling_policy == "fixed":
        min_in_len = [args.fixed_prompt_len] * args.num_requests
        max_in_len = min_in_len
        min_out_len = [args.fixed_output_len] * args.num_requests
        max_out_len = min_out_len
    elif args.sampling_policy == "normal":
        min_in_len = np.rint(np.random.normal(args.prompt_len_mean, args.prompt_len_std, size=args.num_requests)).astype(np.int32)
        max_in_len = min_in_len
        min_out_len = np.rint(np.random.normal(args.output_len_mean, args.output_len_std, size=args.num_requests)).astype(np.int32)
        max_out_len = min_out_len
        
    if args.sampling_policy != "undefined" and args.sampling_policy != "order" and min_in_len is None:
        raise RuntimeError("Invalid input length and output length")
    if args.tokenizer is None:
        tokenizer = None
        args.add_stream_usage = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # load dataset
    if "same_input" in args.dataset:
        samples = [(make_placeholder_prompt(args.fixed_prompt_len, tokenizer=tokenizer), args.fixed_prompt_len, args.fixed_output_len)] * args.num_requests
    elif "sharegpt" in args.dataset.lower():
        filename, file_extension = os.path.splitext(args.dataset)
        if os.path.isabs(args.dataset) and os.path.exists(args.dataset) and file_extension.lower() == ".json":
            samples = util.load_requests_from_json(tokenizer, args.dataset, args.num_requests, min_in_len, max_in_len, min_out_len, max_out_len)
        else:
            dataset = util.load_sharegpt_dataset(args.dataset)
            samples = util.filter_samples_from_dataset(dataset, tokenizer, args.num_requests, min_in_len, max_in_len, min_out_len, max_out_len)
    logger.info(f"Got {len(samples)} requests")
    if len(samples) == 0:
        raise RuntimeError(f"Failed to load samples from dataset: {args.dataset}")
    while len(samples) < args.num_requests:
        samples.append(samples[random.randint(0, len(samples)-1)])
    contexts = []
    for i in range(len(samples)):
        d = samples[i]
        if i < PRINT_SAMPLES and args.verbose:
            logger.info(f"Request[{i}]: {d[1]} / {d[2]}, {d[0][0: 100]}")
        contexts.append(Context(index=i, prompt=d[0], prompt_len=d[1], max_tokens=d[2]))

    # send requests async
    extra = {}
    ignore_eos = False if args.disable_ignore_eos else True
    sender = AysncRequestSender(args.endpoint, args.model, args.api_key, SYS_PROMPT if args.add_system_prompt else None, False if args.disable_stream else True, args.add_stream_usage, ignore_eos, args.verbose)
    
    if not args.no_warmup:
        logger.info("Warmup")
        start_time = time.perf_counter()
        asyncio.run(sender.post_batch_requests_async(contexts[0:2], args.api_kind == "chat", 2, extra))
        end_time = time.perf_counter()
        logger.info(f"Warmup fininshed in {end_time - start_time} seconds")
    for i in range(min(2, len(contexts))):
        contexts[i].clean()

    logger.info("Benchmark")
    if args.no_ramp_up:
        # Original behavior: send all requests at full concurrency
        start_time = time.perf_counter()
        asyncio.run(sender.post_batch_requests_async(contexts, args.api_kind == "chat", args.parallel, extra))
        e2e_duration = time.perf_counter() - start_time
        logger.info(f"Benchmark fininshed in {e2e_duration} seconds")
        measure_contexts = contexts
    else:
        # Ramp-up mode: add ramp-up and tail requests to maintain steady-state concurrency
        ramp_up_count = args.parallel
        tail_count = args.parallel
        total_needed = ramp_up_count + args.num_requests + tail_count
        logger.info(f"Ramp-up mode: ramp_up={ramp_up_count}, measurement={args.num_requests}, tail={tail_count}, total={total_needed}")

        # Prepare extra contexts for ramp-up and tail by duplicating from existing samples
        all_contexts = []
        for i in range(total_needed):
            src = contexts[i % len(contexts)]
            if i < len(contexts):
                all_contexts.append(src)
            else:
                all_contexts.append(Context(index=i, prompt=src.prompt, prompt_len=src.prompt_len, max_tokens=src.max_tokens))

        start_time = time.perf_counter()
        asyncio.run(sender.post_batch_requests_with_rampup(all_contexts, args.api_kind == "chat", args.parallel, extra, ramp_up_count))
        total_duration = time.perf_counter() - start_time
        logger.info(f"Benchmark (total including ramp-up/tail) fininshed in {total_duration} seconds")

        # Only measure the middle num_requests contexts
        measure_contexts = all_contexts[ramp_up_count : ramp_up_count + args.num_requests]

        # Calculate duration from measurement window using absolute timestamps
        valid_measure = [ctx for ctx in measure_contexts if not ctx.error and ctx.request_start_time > 0]
        if valid_measure:
            measure_start = min(ctx.request_start_time for ctx in valid_measure)
            measure_end = max(ctx.request_start_time + ctx.e2e_latency for ctx in valid_measure)
            e2e_duration = measure_end - measure_start
        else:
            e2e_duration = total_duration
        logger.info(f"Measurement window duration: {e2e_duration:.2f} seconds (out of {total_duration:.2f} total)")

    # metrics
    metrics, metrics_good = calculate_metrics(tokenizer, measure_contexts, e2e_duration, args.slo_ttft, args.slo_tpot)
    if metrics is None:
        logger.warning("Failed to get metrics")
        return
    for ctx in measure_contexts:
       if ctx.error:
            logger.warning(f"[{ctx.index}] ERROR: {ctx.error}")
       if not args.disable_warn_dismatch_output_len and ctx.max_tokens > 0 and abs(ctx.output_len - ctx.max_tokens) > 10:
            logger.warning(f"[{ctx.index}] Mismatched output length: expected {ctx.max_tokens}, got {ctx.output_len}")
    e2e_latency_p, ttft_p, tpot_p, tps_p = metrics.get_percentile([50, 90, 99])
    e2e_latency_avg, ttft_avg, tpot_avg, tps_avg = metrics.get_avg()

    prompt_len, gen_len = metrics.input_tokens / len(metrics.ttft), metrics.output_tokens / len(metrics.ttft)
    # if "llama-70b-completions_online_dataset" in args.dataset or "csv" in args.dataset:
    #     prompt_len = sum(sample[1] for sample in samples) / len(samples)
    #     gen_len = sum(sample[2] for sample in samples) / len(samples)
    # elif args.sampling_policy == "fixed":
    #     prompt_len, gen_len = args.fixed_prompt_len, args.fixed_output_len
    # elif args.sampling_policy == "normal":
    #     prompt_len, gen_len = args.prompt_len_mean, args.output_len_mean
    output = f"\n[BeginMetrics] {datetime.now().strftime('%m%d:%H-%M')}\n"
    output += f"log: {args.log_file}\n"
    output += f"model: {args.model}\n"
    output += f"sampling-policy: {args.sampling_policy}\n"
    output += f"sequence-length: {prompt_len}, {gen_len}\n"
    output += f"num-requests: {args.num_requests}\n"
    if not args.no_ramp_up:
        output += f"ramp-up: enabled (ramp={args.parallel}, tail={args.parallel}, total-sent={args.parallel + args.num_requests + args.parallel})\n"
    output += f"batch-size: {args.parallel}\n"
    output += f"e2e-latency(avg, P50, P90, P99): {e2e_latency_avg:0.2f}, {e2e_latency_p[0]:.2f}, {e2e_latency_p[1]:.2f}, {e2e_latency_p[2]:.2f}\n"
    output += f"ttft(avg, P50, P90, P99): {ttft_avg:.2f}, {ttft_p[0]:.2f}, {ttft_p[1]:.2f}, {ttft_p[2]:.2f}\n"
    output += f"tpot(avg, P50, P90, P99): {tpot_avg:.2f}, {tpot_p[0]:.3f}, {tpot_p[1]:.3f}, {tpot_p[2]:.3f}\n"
    output += f"tps(avg, P50, P90, P99): {tps_avg:.2f}, {tps_p[0]:.1f}, {tps_p[1]:.1f}, {tps_p[2]:.1f}\n"
    output += f"throughput: {metrics.input_tokens/e2e_duration:.2f}, {metrics.output_tokens/e2e_duration:.2f}, {(metrics.input_tokens+metrics.output_tokens)/e2e_duration:.2f}\n"
    output += f"rps: {len(measure_contexts)/e2e_duration:.3f}\n"
    output += f"goodput-throughput: {metrics_good.output_tokens/e2e_duration:.2f}\n"
    output += f"goodput-rps: {len(metrics_good.ttft)/e2e_duration:.3f}\n"
    output += f"e2e_duration: {e2e_duration}\n"
    output += f"errors: {len(metrics.errors)}\n"
    if args.record_raw_metrics:
        output += f"raw-ttft: {metrics.ttft}\n"
        output += f"raw-tpot: {metrics.tpot}\n"
    output += f"[EndMetrics]\n"
    print(output)
    if args.log_file:
        with open(args.log_file, "a") as f:
            f.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    # LLM Server
    parser.add_argument("--endpoint", type=str, help="The LLM serving endpoint, for example: http://localhost:18011/v1, or http://localhost:8000/v2/models/ensemble")
    parser.add_argument("--backend", type=str, default="vllm", help="The backend e.g. vllm, trtllm, default is vllm")
    parser.add_argument("--api-key", type=str, help="The api key to call commercial inference API")
    parser.add_argument("--api-kind", type=str, default="completions", choices=["chat", "completions"], help="Can be: chat or completions(default)")
    # input parameters
    parser.add_argument("--model", type=str, help="The model name, if not set, call 'endpoint/models' to query")
    # test data sampling
    parser.add_argument("--sampling-policy", type=str, default="nature", choices=["nature", "fixed", "normal", "undefined", "order"])
    parser.add_argument("--min-prompt-len", type=int, default=4)
    parser.add_argument("--min-output-len", type=int, default=4)
    parser.add_argument("--max-prompt-len", type=int, default=4096)
    parser.add_argument("--max-output-len", type=int, default=4096)
    parser.add_argument("--fixed-prompt-len", type=int, default=3500)
    parser.add_argument("--fixed-output-len", type=int, default=500)
    parser.add_argument("--prompt-len-mean", type=int, default=550)
    parser.add_argument("--prompt-len-std", type=int, default=150)
    parser.add_argument("--output-len-mean", type=int, default=150)
    parser.add_argument("--output-len-std", type=int, default=20)
    # press test setting
    parser.add_argument("--no-warmup", action="store_true", help="Disable warmup")
    parser.add_argument("--no-ramp-up", action="store_true", help="Disable gradual ramp-up/tail mechanism. When ramp-up is enabled (default), extra requests are sent before and after the measurement window to ensure steady-state concurrency.")
    parser.add_argument("--num-requests", type=int, default=1000, help="Number of prompts for benckmark.")
    parser.add_argument("--parallel", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="sharegpt", help="The local folder path to the dataset for testing")
    parser.add_argument("--tokenizer", type=str, help="The local folder path to the model data for token decoding and encoding")
    parser.add_argument("--add-system-prompt", action="store_true", help="add system prompt in front of each conversation")
    parser.add_argument("--disable-stream", action="store_true", help="Disable stream mode")
    parser.add_argument("--disable-ignore-eos", action="store_true", help="Ignore EOS of the output")
    parser.add_argument("--disable-warn-dismatch-output-len", action="store_true", help="warn when generated tokens number is not equal to expected output_len")
    parser.add_argument("--add-stream-usage", action="store_true", help="include stream usage in the request")
    # log
    parser.add_argument("--log-file", type=str, help="file to save log information")
    parser.add_argument("--record-raw-metrics", action="store_true", help="Dump raw metrics like TTFT or TPOT")
    parser.add_argument("--verbose", action="store_true", help="print in verbose mode")
    parser.add_argument("--max-sample-request-tokens", type=int, default=16384, help="set max sample requests tokens")
    parser.add_argument("--slo-ttft", type=float, default=2, help="the sla-ttft milliseconds used to calculate the goodput")
    parser.add_argument("--slo-tpot", type=float, default=0.08, help="the sla-ttft milliseconds used to calculate the goodput")

    args = parser.parse_args()

    # set_ulimit: target_soft_limit=65535
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)
    if current_soft < 65535:
        try:
            resource.setrlimit(resource_type, (65535, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")
    main(args)

