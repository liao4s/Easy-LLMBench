import requests
import aiohttp
import argparse
import asyncio
import json
import numpy as np
import random
import subprocess
import time
import copy

from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from loguru import logger
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from typing import List, Tuple, Union, Optional, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

PRINT_GENERATE_TEXT=False
STRICT_SEMAPHORE=False
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

@dataclass
class Metrics:
    duration: float = field(default=0)
    errors: List[str] = field(default_factory=list)
    e2e_latency: List[float] = field(default_factory=list)
    ttft: List[float] = field(default_factory=list)
    tpot: List[float] = field(default_factory=list)
    tps: List[float] = field(default_factory=list)
    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)

    def get_percentile(self, percent):
        return np.percentile(self.e2e_latency, percent), np.percentile(self.ttft, percent), np.percentile(self.tpot, percent), np.percentile(self.tps, percent)

    def get_avg(self):
        return np.mean(self.e2e_latency), np.mean(self.ttft), np.mean(self.tpot), np.mean(self.tps)

@dataclass
class Context:
    ## input
    index: int = field(default=0)
    prompt: str = field(default="")
    prompt_len: int = field(default=0)
    max_tokens: int = field(default=0)
    ## outpt
    error: Optional[str] = field(default=None)
    generated: str = field(default="")
    output_len: int = field(default=0)
    prompt_tokens: int = field(default=0)
    e2e_latency: float = field(default=0)
    decode_latency: float = field(default=0)
    ttft: Optional[float] = field(default=None)
    tpot: Optional[float] = field(default=None)
    tps: float = field(default=0)

    def clean(self):
        self.error = None
        self.generated = ""
        self.e2e_latency = 0
        self.decode_latency = 0
        self.ttft = None
        self.tpot = None
        self.tps = 0

@dataclass
class InputParameter:
    model: str = field(default="")
    prompt: Optional[str] = field(default=None)
    messages: Optional[List[Dict[str, str]]] = field(default=None)
    n: Optional[int] = field(default=None)
    best_of: Optional[int] = field(default=None)
    use_beam_search: Optional[bool] = field(default=None)
    presence_penalty: Optional[float] = field(default=None)
    frequency_penalty: Optional[float] = field(default=None)
    repetition_penalty: Optional[float] = field(default=None)
    temperature: Optional[float] = field(default=None)
    top_p: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    max_tokens: Optional[int] = field(default=None)
    ignore_eos: Optional[bool] = field(default=None)
    stream: Optional[bool] = field(default=None)

    def to_dict(self):
        output = {}
        for k in self.__dict__:
            if self.__dict__[k] is not None:
                output[k] = self.__dict__[k]
        return output


class AysncRequestSender:
    def __init__(self, endpoint: str, model: str, api_key: Optional[str], sys_prompt: Optional[str], stream: Optional[bool], stream_usage: Optional[bool], ignore_eos: Optional[bool], verbose: Optional[bool]):
        self.endpoint = endpoint
        self.model = model
        self.headers = OrderedDict({"Content-Type": "application/json"})
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.sys_prompt = sys_prompt
        self.stream = stream
        self.stream_usage = stream_usage
        self.ignore_eos = ignore_eos
        self.verbose = verbose

    async def _update_sem(self, sem: asyncio.Semaphore, update_interval: int, ramp_up_period: int, batch_size: int, num_requests: int):
        n_parts = ramp_up_period // update_interval
        base = batch_size // n_parts
        remainder = batch_size % n_parts
        partitions = [base] * (n_parts - remainder) + [base + 1] * remainder
        for p in partitions:
            if len(self.responses) >= num_requests:
                break
            await asyncio.sleep(update_interval)
            for _ in range(p):
                sem.release()

    async def post_batch_requests_async(self, contexts: List[Context], chat: bool, parallel: int, extra: Dict[str, Any]):
        tasks: List[asyncio.Task] = []
        num = len(contexts)
        progress_bar = async_tqdm(total=num,  desc="Processing Requests", smoothing=0.0)
        ## Create a single task to update semaphore
        if parallel is None or parallel <= 0:
            parallel = 1000
        semaphore = asyncio.Semaphore(parallel)
        if STRICT_SEMAPHORE:
            for _ in range(len(contexts) - 1):
                await semaphore.acquire()
            tasks.append(asyncio.create_task(self._update_sem(semaphore, 1, 60, parallel - 1, num)))
        for i in range(num):
            task = asyncio.create_task(self._post_one_request_with_semaphore(semaphore, contexts[i], chat, extra))
            tasks.append(task)
            task.add_done_callback(lambda _: progress_bar.update())
        await asyncio.gather(*tasks)
        progress_bar.close()

    async def _post_one_request_with_semaphore(self, semaphore: asyncio.Semaphore, ctx: Context, chat: bool, extra: Dict[str, Any]):
        async with semaphore:
            if "models/ensemble" in self.endpoint:
                url = self.endpoint + "/generate_stream" if self.stream else self.endpoint + "/generate"
            else:
                url = self.endpoint+"/chat/completions" if chat else self.endpoint+"/completions"
            await self.do_post_request(url, ctx, chat, extra)

    async def do_post_request(self, url: str, ctx: Context, chat: bool, extra: Dict[str, Any]):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            is_trtllm = "models/ensemble" in url
            ## prepare header, payload and params
            payload = {
                "model": self.model,
                "temperature": 0.8,
                "top_p": 1.0,
                "best_of": 1,
                "nvext": {"ignore_eos": self.ignore_eos},
                **extra,
            }
            if self.stream is not None:
                payload["stream"] = self.stream
                if self.stream_usage is not None and self.stream_usage:
                    payload["stream_options"] = {"include_usage": True}
            if self.ignore_eos is not None:
                payload["ignore_eos"] = self.ignore_eos
                payload["nvext"] = {"ignore_eos": self.ignore_eos}
            if ctx.max_tokens > 0:
                payload["max_tokens"] = ctx.max_tokens
            else:
                payload["ignore_eos"] = False ## do NOT ignore eos if max_tokens is not specified
            if chat:
                payload["messages"] = []
                if self.sys_prompt is not None:
                    payload["messages"].append({"role": "system", "content": self.sys_prompt})
                payload["messages"].append({"role": "user", "content": ctx.prompt})
            else:
                key = "text_input" if is_trtllm else "prompt"
                if self.sys_prompt is not None:
                    payload[key] = f"<s>[INST] <<SYS>>\n{self.sys_prompt}<</SYS>>\n\n{ctx.prompt} [/INST]"
                else:
                    payload[key] = f"<s>[INST] {ctx.prompt} [/INST]"
            if self.verbose and ctx.index == 0:
                logger.info(f"URL: {url}, payload: {payload}")
            ## post now
            if self.verbose:
                logger.info(f"Send request[{ctx.index}], {ctx.prompt_len}, {ctx.max_tokens}")
            request_start_time = time.perf_counter()
            # print(f"[AKing info] ignore_eos: {payload['ignore_eos']}, nvext: {payload['nvext']}, max_tokens: {payload['max_tokens']}")
            #print(f"=== begin post: {url}, {payload}")
            async with session.post(url, headers = self.headers, json = payload) as res:
                if res.status != 200:
                    text = await res.text()
                    ctx.error = f"{res.status}--{res.reason}: {text}"
                else:
                    async for chunk_bytes in res.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        try:
                            chunk = chunk_bytes.decode("utf-8")
                            if chunk.startswith("data: "):
                                chunk = chunk[6:]
                            #print(f"==== chunk: ++{chunk}++")
                            if chunk == ": OPENROUTER PROCESSING":
                                continue
                            if chunk == "[DONE]":
                                ctx.e2e_latency = time.perf_counter() - request_start_time
                            else:
                                obj = json.loads(chunk)
                                content = None
                                if "choices" in obj and len(obj["choices"]) > 0:
                                    choice0 = obj["choices"][0]
                                    if "text" in choice0: ## completions API
                                        content = choice0["text"]
                                    elif "delta" in choice0: ## chat-completions API
                                        #if "role" in choice0["delta"] and choice0["delta"]["role"] == "assistant" and "content" in choice0["delta"]:
                                        if "content" in choice0["delta"]:
                                            content = choice0["delta"]["content"]
                                elif "text_output" in obj:
                                    content = obj["text_output"]
                                elif "usage" in obj and obj["usage"] is not None:
                                    if "completion_tokens" in obj["usage"]:
                                        ctx.output_len = obj["usage"]["completion_tokens"]
                                    if "prompt_tokens" in obj["usage"]:
                                        ctx.prompt_tokens = obj["usage"]["prompt_tokens"]
                                if ctx.ttft is None:
                                    ctx.ttft = time.perf_counter() - request_start_time
                                if content is not None:
                                    ## print
                                    #print(content, end="", flush=True)
                                    ctx.generated += content
                        except json.decoder.JSONDecodeError as err:
                            logger.warning(f"Failed to load json string: {chunk}, error: {err}")
                            break
                        except Exception as err:
                            logger.warning(f"Failed to handle streaming chunk: {res.status}, error: {err}")
                            break

            if ctx.e2e_latency < 0.0001:
                ctx.e2e_latency = time.perf_counter() - request_start_time
            if self.verbose:
                logger.info(f"Finished request[{ctx.index}], duration: {ctx.e2e_latency}")

def calculate_metrics(tokenizer: AutoTokenizer, contexts: List[Context], duration: float, slo_ttft: float = 2, slo_tpot: float = 0.08) -> Optional[Metrics]:
    if len(contexts) == 0:
        return None
    m_good = Metrics()
    m_good.duration = duration
    m = Metrics()
    m.duration = duration
    for i in range(len(contexts)):
        ctx = contexts[i]
        if ctx.error:
            m.errors.append(ctx.error)
            continue
        if ctx.prompt_len == 0:
            ctx.prompt_len = ctx.prompt_tokens
        if ctx.output_len == 0 and tokenizer is not None:
            ctx.output_len = len(tokenizer.encode(ctx.generated))
        if ctx.output_len == 0:
            ctx.error = f"Request {ctx.index} has empty output"
            m.errors.append(ctx.error)
            m_good.errors.append(ctx.error)
            continue
        if ctx.ttft is None:
            ctx.error = f"Request {ctx.index} has invalid ttft"
            m.errors.append(ctx.error)
            m_good.errors.append(ctx.error)
            continue
        ctx.decode_latency = ctx.e2e_latency - ctx.ttft
        ctx.tpot = ctx.decode_latency / ctx.output_len
        ctx.tps = ctx.output_len / ctx.decode_latency
        if ctx.ttft < slo_ttft and ctx.tpot < slo_tpot:
            m_good.input_tokens += ctx.prompt_len
            m_good.output_tokens += ctx.output_len
            m_good.e2e_latency.append(ctx.e2e_latency)
            m_good.ttft.append(ctx.ttft)
            m_good.tpot.append(ctx.tpot)
            m_good.tps.append(ctx.tps)
        m.input_tokens += ctx.prompt_len
        m.output_tokens += ctx.output_len
        m.e2e_latency.append(ctx.e2e_latency)
        m.ttft.append(ctx.ttft)
        m.tpot.append(ctx.tpot)
        m.tps.append(ctx.tps)
    return m, m_good