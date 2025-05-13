# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

client = OpenAI(base_url="http://localhost:5678", api_key="foo")
#model="mistralai/Mistral-7B-Instruct-v0.3"
#model="Qwen/Qwen2.5-14B-Instruct"
model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

def make_messages(pad_len, uuid=False):
    pad = "O" + "o" * pad_len + "mmmh"
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": (
                "First, I'm going to let some steam"
                f" off by engaging in Yoga ... {pad}."
                " Ah, I feel better. Now would you please"
                " write me an essay on how meditation can"
                " affect my life?"
            )
        },
    ]

def complete(messages, **kwargs):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        extra_body={
            **kwargs,
        },
    )

def prompt_len(messages):
    return complete(messages, max_tokens=1).usage.prompt_tokens

def output_len(messages, max_tokens):
    out = complete(
        messages,
        max_tokens=max_tokens,
        min_tokens=max_tokens,
    )
    # print(out.choices[0].message.content + "\n", end="")
    return out.usage.completion_tokens

def find_pad_len():
    lo, hi = 0, (args.prompt * 10)
    ntoks = prompt_len(make_messages(hi))
    while True:
        if hi - lo <= 1:
            pad_len = hi
            break
        mid = (lo + hi) // 2
        tk = prompt_len(make_messages(mid))
        if tk == args.prompt:
            pad_len = mid
            ntoks = tk
            break
        if tk < args.prompt:
            lo = mid
        else:
            hi = mid
            ntoks = tk
    return pad_len, ntoks

def run_benchmark(pad_len):
    print(f"starting {args.calls} generations")
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.calls) as ex:
        lens = list(ex.map(
            lambda _: output_len(make_messages(pad_len), args.decode),
            range(args.calls)
        ))
    gen_time = time.monotonic() - t0
    tps = sum(lens) / gen_time
    return sum(lens), gen_time, tps

def main(pad_len):
    all_tps = []
    for _ in range(args.best_of):
        ntoks, gen_time, tps = run_benchmark(pad_len)
        print(f"generated {ntoks} tokens in {gen_time:.1f}s, {tps:.2f}toks/s")
        all_tps.append(tps)
    
    if args.best_of > 1:
        print(f"best of {args.best_of}: {max(all_tps):.2f}toks/s")
    
    return max(all_tps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm", type=str)
    parser.add_argument("--sweep", type=str)
    parser.add_argument("--sweep-steps", type=int, default=10)
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--calls", type=int, default=512)
    parser.add_argument("--prompt", type=int, default=1024)
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--min-decode-ratio", type=float, default=1/64)
    parser.add_argument("--max-decode-ratio", type=float, default=2)
    args = parser.parse_args()

    if args.vllm is not None:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="foo")
        model = args.vllm

    print(f"finding pad length for input of size {args.prompt}...")
    pad_len, ntoks = find_pad_len()
    # ntoks = pad_len = 42
    print(f"... pad length must be {pad_len}")
    print(f"... prompt length is {ntoks}")

    if args.sweep:
        min_dr = math.log(args.min_decode_ratio)
        max_dr = math.log(args.max_decode_ratio)
        ratios = [
            min_dr + i / (args.sweep_steps-1) * (max_dr - min_dr)
            for i in range(args.sweep_steps)
        ]
        entries = []
        for ratio in map(math.exp, ratios):
            args.decode = round(args.prompt * ratio)
            print(f"... ratio {ratio} (decode={args.decode})")
            tps = main(pad_len)
            entries.append({
                "prompt": args.prompt,
                "decode": args.decode,
                "ratio": ratio,
                "tps": tps,
            })
        with open(args.sweep, "a") as f:
            f.write(json.dumps(entries) + "\n")
        
    else:
        main(pad_len)
