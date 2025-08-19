# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
r"""Benchmark online serving throughput.
On the server side, run one of the following commands:
    vLLM OpenAI API server
    vllm serve <your_model> \
        --swap-space 16 \
        --disable-log-requests
On the client side, run:
    python benchmarks/benchmark_serving_embedding.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000
    when using tgi backend, add
        --endpoint /generate_stream
    to the end of the command above.
"""

import argparse
import asyncio
import gc
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Optional

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from backend_request_func import (
    ASYNC_EMBEDDING_REQUEST_FUNCS,
    EmbeddingRequestFuncInput,
    EmbeddingRequestFuncOutput,
)

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from argparse import ArgumentParser as FlexibleArgumentParser

from benchmark_dataset import (
    EmbeddingRandomDataset,
    EmbeddingSampleRequest,
)
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json
from vllm.benchmarks.serve import get_request

MILLISECONDS_TO_SECONDS_CONVERSION = 1000


@dataclass
class EmbeddingBenchmarkMetrics:
    completed: int
    total_input: int
    request_throughput: float
    total_token_throughput: float
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]


def calculate_embedding_metrics(
    input_requests: list[str],
    outputs: list[EmbeddingRequestFuncOutput],
    dur_s: float,
    selected_percentiles: list[float],
    batch_size: int,
) -> EmbeddingBenchmarkMetrics:
    total_input = 0
    completed = 0
    e2els: list[float] = []
    for i in range(len(outputs)):
        if outputs[i].success:
            total_input += input_requests[i].prompt_len * batch_size
            e2els.append(outputs[i].latency)
            completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = EmbeddingBenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        request_throughput=completed / dur_s,
        total_token_throughput=(total_input) / dur_s,
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
    )

    return metrics


async def benchmark(
    backend: str,
    api_url: str,
    base_url: str,
    model_id: str,
    model_name: str,
    # tokenizer: PreTrainedTokenizerBase,
    input_requests: list[EmbeddingSampleRequest],
    request_rate: float,
    burstiness: float,
    disable_tqdm: bool,
    profile: bool,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    batch_size: int,
    max_concurrency: Optional[int],
    ramp_up_strategy: Optional[Literal["linear", "exponential"]] = None,
    ramp_up_start_rps: Optional[int] = None,
    ramp_up_end_rps: Optional[int] = None,
):
    if backend in ASYNC_EMBEDDING_REQUEST_FUNCS:
        request_func = ASYNC_EMBEDDING_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print("Starting initial single prompt test run...")
    test_prompt, test_prompt_len = (
        input_requests[0].prompt,
        input_requests[0].prompt_len,
    )

    test_input = EmbeddingRequestFuncInput(
        model=model_id,
        model_name=model_name,
        prompt=test_prompt,
        batch_size=batch_size,
        api_url=api_url,
        prompt_len=test_prompt_len,
    )

    test_output = await request_func(request_func_input=test_input)
    if not test_output.success:
        raise ValueError(
            "Initial test run failed - Please make sure benchmark arguments "
            f"are correctly specified. Error: {test_output.error}"
        )
    else:
        print("Initial test run completed. Starting main benchmark run...")

    # TODO: double check
    if profile:
        print("Starting profiler...")
        profile_input = EmbeddingRequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_prompt,
            api_url=base_url + "/start_profile",
            prompt_len=test_prompt_len,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler started")

    distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"

    if ramp_up_strategy is not None:
        print(
            f"Traffic ramp-up strategy: {ramp_up_strategy}. Will increase "
            f"RPS from {ramp_up_start_rps} to {ramp_up_end_rps} RPS over "
            "the duration of the benchmark."
        )
    else:
        print(f"Traffic request rate: {request_rate} RPS.")

    print(f"Burstiness factor: {burstiness} ({distribution})")
    print(f"Maximum request concurrency: {max_concurrency}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    # This can be used once the minimum Python version is 3.10 or higher,
    # and it will simplify the code in limited_request_func.
    #    semaphore = (asyncio.Semaphore(max_concurrency)
    #                 if max_concurrency else contextlib.nullcontext())
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_func_input, pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_func_input, pbar=pbar)

    benchmark_start_time = time.perf_counter()
    tasks: list[asyncio.Task] = []

    rps_change_events = []
    last_int_rps = -1
    if ramp_up_strategy is not None and ramp_up_start_rps is not None:
        last_int_rps = ramp_up_start_rps
        rps_change_events.append(
            {
                "rps": last_int_rps,
                "timestamp": datetime.now().isoformat(),
            }
        )

    async for request, current_request_rate in get_request(
        input_requests,
        request_rate,
        burstiness,
        ramp_up_strategy,
        ramp_up_start_rps,
        ramp_up_end_rps,
    ):
        if ramp_up_strategy is not None:
            current_int_rps = int(current_request_rate)
            if current_int_rps > last_int_rps:
                timestamp = datetime.now().isoformat()
                for rps_val in range(last_int_rps + 1, current_int_rps + 1):
                    rps_change_events.append({"rps": rps_val, "timestamp": timestamp})
                last_int_rps = current_int_rps

        prompt, prompt_len = (
            request.prompt,
            request.prompt_len,
        )
        req_model_id, req_model_name = model_id, model_name

        request_func_input = EmbeddingRequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            batch_size=batch_size,
        )
        task = limited_request_func(request_func_input=request_func_input, pbar=pbar)
        tasks.append(asyncio.create_task(task))
    outputs: list[EmbeddingRequestFuncOutput] = await asyncio.gather(*tasks)
    for o in outputs:
        print(f"{o.latency=}")

    # TODO Double check
    if profile:
        print("Stopping profiler...")
        profile_input = EmbeddingRequestFuncInput(
            model=model_id,
            prompt=test_prompt,
            api_url=base_url + "/stop_profile",  # double check
            prompt_len=test_prompt_len,
        )
        profile_output = await request_func(request_func_input=profile_input)
        if profile_output.success:
            print("Profiler stopped")

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics = calculate_embedding_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        # TODO: do not need this version
        # tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        batch_size=batch_size,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Batch size:", batch_size))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "request_throughput": metrics.request_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "errors": [output.error for output in outputs],
    }

    if rps_change_events:
        result["rps_change_events"] = rps_change_events

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result


def save_to_pytorch_benchmark_format(
    args: argparse.Namespace, results: dict[str, Any], file_name: str
) -> None:
    # TODO: check this later
    metrics = []
    # These raw data might be useful, but they are rather big. They can be added
    # later if needed
    ignored_metrics = ["errors"]
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={k: [results[k]] for k in metrics},
        extra_info={
            k: results[k]
            for k in results
            if k not in metrics and k not in ignored_metrics
        },
    )
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(file_name)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    model_name = args.served_model_name
    # tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
    # tokenizer_mode = args.tokenizer_mode
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    # double check
    distribution = args.distribution
    num_prompts = args.num_prompts
    prompt_length = args.prompt_length

    # Validate ramp-up arguments
    if args.ramp_up_strategy is not None:
        if args.request_rate != float("inf"):
            raise ValueError(
                "When using ramp-up, do not specify --request-rate. "
                "The request rate will be controlled by ramp-up parameters. "
                "Please remove the --request-rate argument."
            )
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError(
                "When using --ramp-up-strategy, both --ramp-up-start-rps and "
                "--ramp-up-end-rps must be specified"
            )
        if args.ramp_up_start_rps < 0 or args.ramp_up_end_rps < 0:
            raise ValueError("Ramp-up start and end RPS must be non-negative")
        if args.ramp_up_start_rps > args.ramp_up_end_rps:
            raise ValueError("Ramp-up start RPS must be less than end RPS")
        if args.ramp_up_strategy == "exponential" and args.ramp_up_start_rps == 0:
            raise ValueError("For exponential ramp-up, the start RPS cannot be 0.")

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
        base_url = f"{args.base_url}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        base_url = f"http://{args.host}:{args.port}"

    # TODO: Add this when we enable
    # tokenizer = get_tokenizer(
        # tokenizer_id,
        # tokenizer_mode=tokenizer_mode,
        # trust_remote_code=args.trust_remote_code,
    # )

    # TODO: Fix the params
    dataset = EmbeddingRandomDataset(
        prompt_length, max(batch_sizes) * num_prompts, distribution
    )

    for batch_size in batch_sizes:
        # Avoid GC processing "static" data - reduce pause times.
        gc.collect()
        gc.freeze()
        # TODO: fix the parameters
        input_requests = dataset.sample(num_prompts, batch_size, distribution)
        benchmark_result = asyncio.run(
            benchmark(
                backend=backend,
                api_url=api_url,
                base_url=base_url,
                model_id=model_id,
                model_name=model_name,
                # tokenizer=tokenizer,
                input_requests=input_requests,
                request_rate=args.request_rate,
                burstiness=args.burstiness,
                disable_tqdm=args.disable_tqdm,
                profile=args.profile,
                selected_percentile_metrics=args.percentile_metrics.split(","),
                selected_percentiles=[
                    float(p) for p in args.metric_percentiles.split(",")
                ],
                batch_size=batch_size,
                max_concurrency=args.max_concurrency,
                ramp_up_strategy=args.ramp_up_strategy,
                ramp_up_start_rps=args.ramp_up_start_rps,
                ramp_up_end_rps=args.ramp_up_end_rps,
            )
        )

        # Save config and results to json
        if args.save_result or args.append_result:
            result_json: dict[str, Any] = {}

            # Setup
            current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_json["date"] = current_dt
            result_json["backend"] = backend
            result_json["model_id"] = model_id
            # result_json["tokenizer_id"] = tokenizer_id
            result_json["num_prompts"] = args.num_prompts

            # Metadata
            if args.metadata:
                for item in args.metadata:
                    if "=" in item:
                        kvstring = item.split("=")
                        result_json[kvstring[0].strip()] = kvstring[1].strip()
                    else:
                        raise ValueError(
                            "Invalid metadata format. Please use KEY=VALUE format."
                        )
            # Traffic
            result_json["request_rate"] = (
                args.request_rate if args.request_rate < float("inf") else "inf"
            )
            # TODO: what is this?
            result_json["burstiness"] = args.burstiness
            result_json["max_concurrency"] = args.max_concurrency

            if args.ramp_up_strategy is not None:
                result_json["ramp_up_strategy"] = args.ramp_up_strategy
                result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
                result_json["ramp_up_end_rps"] = args.ramp_up_end_rps

            # Merge with benchmark result
            result_json = {**result_json, **benchmark_result}

            if not args.save_detailed:
                # Remove fields with too many data points
                for field in [
                    "input_lens",
                    "errors",
                ]:
                    if field in result_json:
                        del result_json[field]
                    if field in benchmark_result:
                        del benchmark_result[field]

            # Save to file
            base_model_id = model_id.split("/")[-1]
            max_concurrency_str = (
                f"-concurrency{args.max_concurrency}"
                if args.max_concurrency is not None
                else ""
            )
            if args.ramp_up_strategy is not None:
                file_name = f"{backend}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
            else:
                file_name = f"{backend}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-{current_dt}.json"  # noqa
            if args.result_filename:
                file_name = args.result_filename
            if args.result_dir:
                os.makedirs(args.result_dir, exist_ok=True)
                file_name = os.path.join(args.result_dir, file_name)
            with open(
                file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
            ) as outfile:
                # Append a newline.
                if args.append_result and outfile.tell() != 0:
                    outfile.write("\n")
                json.dump(result_json, outfile)
            save_to_pytorch_benchmark_format(args, result_json, file_name)


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark the online serving throughput for embeddings api."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_EMBEDDING_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    # Use 127.0.0.1 here instead of localhost to force the use of ipv4
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/embeddings",
        help="API endpoint.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )
    # TODO: Do not need this for now
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer, if not using the default tokenizer.",  # noqa: E501
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts to process for each batch size.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,16,64,256",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument(
        "--prompt-length", type=int, default=128, help="Length of prompts to encode"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="fixed",
        choices=["uniform", "normal", "fixed"],
        help="Distribution of prompt length for each batch size (default: fixed)",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
    )

    # TODO: Double check
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation. "
        "Only take effect when request_rate is not inf. "
        "Default value is 1, which follows Poisson process. "
        "Otherwise, the request intervals follow a gamma distribution. "
        "A lower burstiness value (0 < burstiness < 1) results in more "
        "bursty requests. A higher burstiness value (burstiness > 1) "
        "results in a more uniform arrival of requests.",
    )
    # TODO: Double check
    parser.add_argument("--seed", type=int, default=0)
    # TODO:
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    # TODO:
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with "
        "VLLM_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="When saving the results, whether to include per request "
        "information such as response, error, ttfs, tpots, etc.",
    )
    parser.add_argument(
        "--append-result",
        action="store_true",
        help="Append the benchmark result to the existing json file.",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "{backend}-{args.request_rate}qps-{base_model_id}-{current_dt}.json"
        " format.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Set ignore_eos flag when sending the benchmark request."
        "Warning: ignore_eos is not supported in deepspeed_mii and tgi.",
    )
    parser.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help="Comma-separated list of selected metrics to report percentils. "
        "This argument specifies the metrics to report percentiles. "
        'Allowed metric names are "ttft", "tpot", "itl", "e2el". '
        'Default value is "ttft,tpot,itl".',
    )
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help="Comma-separated list of percentiles for selected metrics. "
        'To report 25-th, 50-th, and 75-th percentiles, use "25,50,75". '
        'Default value is "99". '
        'Use "--percentile-metrics" to select metrics.',
    )

    # group for dataset specific arguments
    custom_group = parser.add_argument_group("custom dataset options")
    custom_group.add_argument(
        "--custom-output-len",
        type=int,
        default=256,
        help="Number of output tokens per request, used only for custom dataset.",
    )
    custom_group.add_argument(
        "--custom-skip-chat-template",
        action="store_true",
        help="Skip applying chat template to prompt, used only for custom dataset.",
    )

    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request, used only for random sampling.",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, "
        "used only for random sampling. Must be in the range [0, 1) to define "
        "a symmetric sampling range"
        "[length * (1 - range_ratio), length * (1 + range_ratio)].",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help=(
            "Number of fixed prefix tokens before the random context "
            "in a request. "
            "The total input length is the sum of `random-prefix-len` and "
            "a random "
            "context length sampled from [input_len * (1 - range_ratio), "
            "input_len * (1 + range_ratio)]."
        ),
    )

    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=["auto", "slow", "mistral", "custom"],
        help='The tokenizer mode.\n\n* "auto" will use the '
        'fast tokenizer if available.\n* "slow" will '
        "always use the slow tokenizer. \n* "
        '"mistral" will always use the `mistral_common` tokenizer. \n*'
        '"custom" will use --tokenizer to select the preregistered tokenizer.',
    )

    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="The model name used in the API. "
        "If not specified, the model name will be the "
        "same as the ``--model`` argument. ",
    )

    parser.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="The ramp-up strategy. This would be used to "
        "ramp up the request rate from initial RPS to final "
        "RPS rate (specified by --ramp-up-start-rps and --ramp-up-end-rps). "
        "over the duration of the benchmark.",
    )
    parser.add_argument(
        "--ramp-up-start-rps",
        type=int,
        default=None,
        help="The starting request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )
    parser.add_argument(
        "--ramp-up-end-rps",
        type=int,
        default=None,
        help="The ending request rate for ramp-up (RPS). "
        "Needs to be specified when --ramp-up-strategy is used.",
    )

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
