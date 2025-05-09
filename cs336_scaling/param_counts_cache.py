import argparse
import json
import os

from cs336_scaling.constants import VOCAB_SIZE, CONTEXT_LENGTH
from cs336_scaling.model import BasicsTransformerLM


D_MODEL_VALUES = list(range(64, 1024 + 64, 64))
NUM_LAYERS_VALUES = list(range(2, 24 + 1))

D_MODEL_TEST_VALUES = [64]
NUM_LAYERS_TEST_VALUES = [2]


def get_param_counts_key(d_model: int, num_layers: int) -> str:
    return f"d{d_model}_l{num_layers}"


def load_param_counts_cache():
    """Load the param counts cache from the output directory."""
    output_dir = "out/cache"
    output_file = os.path.join(output_dir, "param_counts.json")
    with open(output_file, "r") as f:
        return json.load(f)


def get_cached_param_counts(d_model: int, num_layers: int) -> dict | None:
    """Get the cached param counts for a given d_model and num_layers."""
    cache = load_param_counts_cache()
    key = get_param_counts_key(d_model, num_layers)
    return cache.get(key, None)


def get_param_counts(
    d_model: int, num_layers: int, calculate_if_missing: bool = True
) -> int:
    cached_params = get_cached_param_counts(d_model, num_layers)

    if cached_params is not None:
        return cached_params

    if not calculate_if_missing:
        raise ValueError(
            f"No cached param counts found for d_model={d_model} and num_layers={num_layers}, set calculate_if_missing=True to calculate"
        )

    print(
        f"Warning: No cached param counts found for d_model={d_model} and num_layers={num_layers}, calculating..."
    )

    return calc_param_counts(d_model, num_layers)


def calc_param_counts(d_model: int, num_layers: int) -> dict:
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=d_model // 64,
        d_ff=d_model * 4,
        attn_pdrop=0.1,
        residual_pdrop=0.1,
    )

    total_params = model.get_num_params(non_embedding=False)
    non_embedding_params = model.get_num_params(non_embedding=True)
    embedding_params = total_params - non_embedding_params

    return {
        "d_model": d_model,
        "num_layers": num_layers,
        "total_params": total_params,
        "embedding_params": embedding_params,
        "non_embedding_params": non_embedding_params,
    }


def cache_param_counts(
    output_dir: str,
    output_file: str,
    d_model_values: list[int] = D_MODEL_VALUES,
    num_layers_values: list[int] = NUM_LAYERS_VALUES,
):
    """
    Cache the param counts (total, non-embedding, embedding) for all the configs
    in the D_MODEL_VALUES and NUM_LAYERS_VALUES, saving the results to the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, output_file)

    param_counts_dict = {}

    configs_total = len(d_model_values) * len(num_layers_values)
    configs_done = 0
    print(f"Caching parameter counts for {configs_total} configs...")

    for d_model in d_model_values:
        for num_layers in num_layers_values:
            param_counts = calc_param_counts(d_model, num_layers)
            key = get_param_counts_key(d_model, num_layers)
            param_counts_dict[key] = param_counts

            configs_done += 1
            print(f"Cached param counts for {key} ({configs_done}/{configs_total})")

    with open(output_file, "w") as f:
        json.dump(param_counts_dict, f, indent=4)

    print(f"Parameter counts saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="out/cache")
    parser.add_argument("--outfile", type=str, default="param_counts.json")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        cache_param_counts(
            args.outdir,
            args.outfile,
            d_model_values=D_MODEL_TEST_VALUES,
            num_layers_values=NUM_LAYERS_TEST_VALUES,
        )
    else:
        cache_param_counts(args.outdir, args.outfile)
