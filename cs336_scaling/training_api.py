"""
API run format:

{
    'd_model': 1024,
    'num_layers': 24,
    'num_heads': 16,
    'batch_size': 128,
    'learning_rate': 0.001,
    'train_flops': 10000000000000000,
    'loss': 9.07103488440561
}

"""

import json
import os
import time
import requests
from dotenv import load_dotenv

from cs336_scaling.common import (
    est_non_embedding_params,
    est_total_params,
    get_chinchilla_lr_for_n,
    get_shape_given_n,
)
from cs336_scaling.constants import BATCH_SIZE


load_dotenv()

BASE_URL = "http://hyperturing.stanford.edu:8000"
API_KEY = os.getenv("API_KEY")


def get_loss(
    d_model: int,
    num_layers: int,
    num_heads: int,
    batch_size: int,
    learning_rate: float,
    train_flops: int,
) -> dict[str, float | int]:
    url = f"{BASE_URL}/loss"

    config = {
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_flops": int(train_flops),
        "api_key": API_KEY,
    }

    return requests.get(url, config).json()


def get_total_flops_used() -> int:
    url = f"{BASE_URL}/total_flops_used"
    return requests.get(url, {"api_key": API_KEY}).json()


def get_previous_runs():
    url = f"{BASE_URL}/previous_runs"
    return requests.get(url, {"api_key": API_KEY}).json()


def get_run_key(run: dict) -> str:
    return f"d{run['d_model']}_L{run['num_layers']}_h{run['num_heads']}_b{run['batch_size']}_lr{run['learning_rate']}_C{run['train_flops']}"


def sync_runs():
    runs = get_previous_runs()["previous_runs"]

    with open("out/runs.json", "r") as f:
        runs_dict = json.load(f)

    for i, run in enumerate(runs):
        run["est_n_total"] = est_total_params(run["d_model"], run["num_layers"])
        run["est_n_non_embedding"] = est_non_embedding_params(
            run["d_model"], run["num_layers"]
        )
        run["est_n_embed"] = run["est_n_total"] - run["est_n_non_embedding"]
        run["est_embed_ratio"] = run["est_n_embed"] / run["est_n_total"]
        run["est_tokens"] = run["train_flops"] / (6 * run["est_n_total"])
        run["est_tokens_per_param"] = run["est_tokens"] / run["est_n_total"]

        # Format learning_rate and train_flops in scientific notation
        run["learning_rate"] = float(f"{run['learning_rate']:.6e}")
        run["train_flops"] = float(f"{run['train_flops']:.6e}")

        run_key = get_run_key(run)

        if run_key in runs_dict and runs_dict[run_key].get("sync_time") is not None:
            run["sync_time"] = runs_dict[run_key]["sync_time"]
        else:
            run["sync_time"] = time.time()

        runs_dict[run_key] = run

    with open("out/runs.json", "w") as f:
        json.dump(runs_dict, f)

    return list(runs_dict.values())


def sync_api_state():
    return sync_runs(), get_total_flops_used()


def get_api_state():
    return get_previous_runs()["previous_runs"], get_total_flops_used()


if __name__ == "__main__":
    sync_api_state()
