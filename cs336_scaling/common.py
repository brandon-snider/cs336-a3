from collections.abc import Callable
import itertools

import numpy as np
from cs336_scaling.constants import (
    CHINCHILLA_PARAMS_LAW_A,
    CHINCHILLA_PARAMS_LAW_B,
    CHINCHILLA_LR_LAW_A,
    CHINCHILLA_LR_LAW_B,
    CHINCHILLA_LR_LAW_N_SCALE,
    VOCAB_SIZE,
    FULL_FLOPS_RANGE,
)


def est_total_params(d_model: int, num_layers: int) -> int:
    """Estimates the total number of parameters in a transformer model."""
    return (12 * num_layers * d_model**2) + (2 * VOCAB_SIZE * d_model)


def est_non_embedding_params(d_model: int, num_layers: int) -> int:
    """Estimates the number of non-embedding parameters in a transformer model."""
    return 12 * num_layers * d_model**2


def pick_candidates_around_n(
    n: int, factor: float, n_candidates: int, round_to_int: bool = False
) -> list[float]:
    """
    Pick a list of candidate values logarithmically spaced around `n`.

    The range of candidates will be approximately [`n / factor`, `n * factor`].

    Args:
        - n: The central value.
        - factor: Determines the range [`n / factor`, `n * factor`]. Must be > 1.0.
        - n_candidates: The number of candidates to generate. Must be >= 1.

    Returns:
        A list of floating-point candidate values.
    """
    if factor <= 1.0:
        raise ValueError("factor must be > 1.0")
    if n_candidates < 1:
        raise ValueError("n_candidates must be >= 1")

    # Calculate the log10 bounds for the desired range
    log_n_min = np.log10(n / factor)
    log_n_max = np.log10(n * factor)

    # Generate logarithmically spaced values
    candidates_float = np.logspace(log_n_min, log_n_max, num=n_candidates)

    # Convert to integers, rounding appropriately
    if round_to_int:
        candidates_round = [int(round(c)) for c in candidates_float]
    else:
        candidates_round = [float(c) for c in candidates_float]

    # Ensure uniqueness and sort, although logspace should already sort them
    # Remove duplicates which might occur after rounding, especially for small n or small factor
    unique_candidates = sorted(list(set(candidates_round)))

    # Ensure the central value n is included if it wasn't already after rounding
    if n not in unique_candidates:
        # Find the closest value to n and replace it, to keep n_candidates the same
        closest_val = min(unique_candidates, key=lambda x: abs(x - n))
        unique_candidates[unique_candidates.index(closest_val)] = n
        unique_candidates = sorted(
            unique_candidates
        )  # Re-sort after potential replacement

    return unique_candidates


def get_shape_for_n_custom(
    n_target: int,
    min_aspect_ratio: int | None = None,
    max_aspect_ratio: int | None = None,
    head_dim_ratio: int | None = None,
    n_is_total: bool = False,
    clamp_to_api: bool = True,
) -> tuple[int, int, int, int]:
    """
    Find and return a transformer shape (d, L, h) that gets as close as possible to n_target.

    Args:
        - n_target: target number of parameters
        - min_aspect_ratio: minimum aspect ratio (d/L)
        - max_aspect_ratio: maximum aspect ratio (d/L)
        - head_dim_ratio: force head dimension to be a multiple of this value (d/h)
            - Default is dynamic based on `n_target`
        - n_is_total:
            - If True, `n_target` representsthe total number of parameters,
            - If False, `n_target` represents the number of non-embedding parameters.
        - clamp_to_api: if True, clamp the shape to the API limits

    If `n_is_total` is True:
        Solve 12Ld^2 + 2Vd = n, keeping L in [2, 24], d in [64, 1024], h in [2, 16]
    Otherwise:
        Solve 12Ld^2 = n, keeping L in [2, 24], d in [64, 1024], h in [2, 16]
    """
    closest = None, None, None, None  # d, l, h, n_realized

    d_max = 1024 if clamp_to_api else 4096
    L_max = 24 if clamp_to_api else 96
    h_max = 16 if clamp_to_api else 128
    head_dim_max = 128 if clamp_to_api else 256
    head_dim_min = 16

    # if n_is_total:
    #     min_aspect_ratio = 48 if n_target < 1e8 else 64
    #     head_dim_ratio = 16 if n_target < 1e8 else 64
    # else:
    #     min_aspect_ratio = 16 if n_target < 1e5 else 32 if n_target < 1e6 else 64
    #     head_dim_ratio = 16 if n_target < 1e6 else 32 if n_target < 1e7 else 64

    if min_aspect_ratio is None:
        min_aspect_ratio = 16 if n_target < 1e5 else 32 if n_target < 1e6 else 64

    if max_aspect_ratio is None:
        max_aspect_ratio = 256

    if head_dim_ratio is None:
        head_dim_ratio = (
            16
            if n_target < 1e6
            else 32
            if n_target < 1e7
            else 64
            if n_target < 1e9
            else 128
        )

    for d, L, h in itertools.product(
        range(64, d_max + 1, 8), range(2, L_max + 1), range(2, h_max + 1)
    ):
        aspect_ratio = d / L

        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue

        head_dim = d / h

        if (
            head_dim < head_dim_min
            or head_dim > head_dim_max
            or head_dim % head_dim_ratio != 0
        ):
            continue

        if n_is_total:
            n_realized = 12 * L * d**2 + 2 * VOCAB_SIZE * d
        else:
            n_realized = 12 * L * d**2

        closest_n_realized = closest[3]
        if not closest_n_realized or abs(n_realized - n_target) < abs(
            closest_n_realized - n_target
        ):
            closest = d, L, h, n_realized

    if closest is None:
        raise ValueError(
            f"No valid (d, L) found for the supplied n ({n_target:,}), n_is_total={n_is_total}."
        )

    d, L, h, n_realized = closest
    err = abs(n_realized - n_target)
    err_pct = err * 100 / n_target

    return d, L, h, n_realized, err, err_pct


def get_shape_given_n(
    n: int,
    min_aspect_ratio: int | None = None,
    max_aspect_ratio: int | None = None,
    L_min: int = 2,
    L_max: int = 24,
    head_dim_default: int | None = None,
    h_max: int = 16,
    h_min: int = 2,
    clamp_to_api: bool = True,
) -> tuple[int, int, int, int]:
    """
    Choose a Transformer shape (d, L, h) that gets as close as possible to `n`
    *non-embedding* parameters, while respecting the provided constraints.

    Args:
        n: target *non-embedding parameter count
        min_aspect_ratio: minimum aspect ratio (d/L)
        max_aspect_ratio: maximum aspect ratio (d/L)
        L_min: minimum number of layers
        L_max: maximum number of layers
        d_head_default: default head dimension
        h_max: maximum number of attention heads
        h_min: minimum number of attention heads

    Returns (d, L, h, n_star):
        - d: model width (multiple of 64)
        - L: number of layers
        - h: number of attention heads  (= d // 64)
        - n_star: realised non-embedding parameter count (12 L d²)
    """
    best_err = None
    best_shape = None

    if min_aspect_ratio is None:
        min_aspect_ratio = 16 if n < 1e5 else 32 if n < 1e6 else 64

    if max_aspect_ratio is None:
        max_aspect_ratio = 256

    if head_dim_default is None:
        head_dim_default = 16 if n < 1e6 else 32 if n < 1e7 else 64 if n < 1e9 else 128

    # Scan integer aspect ratios aspect_ratio = d/L in [min_aspect_ratio, max_aspect_ratio]
    for aspect_ratio in range(min_aspect_ratio, max_aspect_ratio + 1):
        # Ideal (real-valued) width for this aspect ratio
        d_ideal = (n * aspect_ratio / 12) ** (1 / 3)

        # Choose d that is closest to ideal, and is a multiple of d_head_default
        d = max(head_dim_default, round(d_ideal / head_dim_default) * head_dim_default)

        if clamp_to_api:
            d = max(64, min(d, 1024))

        # Choose depth that best respects this aspect ratio, and is within bounds
        L = max(L_min, round(d / aspect_ratio))

        if clamp_to_api:
            L = min(L_max, L)

        # Check the rounded aspect ratio is still acceptable
        if not (min_aspect_ratio <= d / L <= max_aspect_ratio):
            continue

        n_star = 12 * L * d**2  # Realised non-embedding parameter count
        err = abs(n_star - n)  # Error in realised n_star from target n

        if best_err is None or err < best_err:
            best_err = err

            # Choose h that is closest to default d_head, and is within bounds
            h = max(h_min, round(d / head_dim_default))

            if clamp_to_api:
                h = min(h_max, h)

            best_shape = (d, L, h, n_star)

    if best_shape is None:  # pathological n?
        print(f"WARNING: No valid (d, L) found for the supplied n ({n}).")
        best_shape = (0, 0, 0, 0)

    return best_shape


def power_law(x: np.ndarray | float, a: float, b: float) -> np.ndarray | float:
    """Power law function: y = a * x^b."""
    return a * np.power(x, b)


def get_part1_n_for_c(c: int) -> int:
    """
    Get the number of parameters for given FLOPs.
    NB: hardcodes the power law from the `chinchilla_isoflops` part of the assignment.
    """
    a = 2.579273e01
    b = 0.403812
    return a * (c**b)


def get_chinchilla_power_law_n_for_c(c: int, tokens_per_param: float = 20) -> int:
    """
    Get the number of parameters for given FLOPs that produces D = N * tokens_per_param, given C = 6ND.
    NB: hardcodes the power law from the Chinchilla paper.
    """
    a = CHINCHILLA_PARAMS_LAW_A
    b = CHINCHILLA_PARAMS_LAW_B
    return int(a * (c**b))


def get_chinchilla_n_for_c(c: int, tokens_per_param: float = 20) -> int:
    """
    Get the number of parameters for given FLOPs that produces:
    D = N * tokens_per_param, given C = 6ND.
    """
    # C = 6N(tokens_per_param * N) = 6N^2 * tokens_per_param
    # => N^2 = C / (6 * tokens_per_param)
    # => N = sqrt(C / (6 * tokens_per_param))
    return int(np.sqrt(c / (6 * tokens_per_param)))


def get_chinchilla_lr_for_n(n: int) -> float:
    """
    Get the Chinchilla-predicted optimal learning rate for given number of parameters.
    NB: hardcodes the power law from the Chinchilla paper.
    """
    a = CHINCHILLA_LR_LAW_A
    b = CHINCHILLA_LR_LAW_B
    return a * (n / CHINCHILLA_LR_LAW_N_SCALE) ** b


def print_predicted_shapes(
    n_for_c_fn: Callable[[int], int] | None = get_chinchilla_n_for_c,
    get_shape_fn=get_shape_given_n,
    get_lr_fn: Callable[[int], float] | None = get_chinchilla_lr_for_n,
    ns: list[int] | None = None,
    c: int | None = None,
):
    if ns is None and n_for_c_fn is None:
        raise ValueError("Must provide either ns or n_for_c_fn")

    if ns is None:
        cs = FULL_FLOPS_RANGE[3:] + [3e19, 6e19, 1e20, 3e20, 6e20, 1e21, 3e21, 6e21, 1e22, 3e22, 6e22, 1e23]  # fmt: skip
        ns = [n_for_c_fn(c) for c in cs]
    else:
        cs = [c] * len(ns) if c is not None else [0] * len(ns)

    # Get the best shape for each N
    shapes = [get_shape_fn(n) for n in ns]

    print("Computed shapes:")

    print(
        f"{'idx':>5}\t{'c':>6}\t{'n':>8}\t{'n_star':>8}\t{'err':>8}\t{'err_pct':>7}\t{'tok/n':>8}\t{'d':>4}\t{'L':>3}\t{'d/L':>5}\t{'h':>3}\t{'pred_lr':>9}\t{'embed_ratio':>9}\t{'tokens':>9}\t{'tok/n_star':>8}"
    )
    for idx, (n, c, shape) in enumerate(zip(ns, cs, shapes)):
        d, L, h, n_star, *_ = shape
        n_embed = 2 * VOCAB_SIZE * d
        n_total = n_star + n_embed
        err = abs(n_star - n)

        if n_total == 0:
            aspect_ratio = 0
            err_pct = 100
            embed_ratio = 0
            tokens = 0
            tokens_per_param = 0
            tokens_per_non_embedding = 0
            lr = 0
        else:
            err_pct = err / n
            aspect_ratio = d / L
            embed_ratio = n_embed / n_total
            tokens = c / (6 * n_total)  # Total dataset size predicted by C = 6ND
            tokens_per_param = tokens / n_total
            tokens_per_non_embedding = tokens / n_star
            lr = get_lr_fn(n_total)

        print(
            f"{idx:>5}\t{c:>6.0e}\t{n:>8.2e}\t{n_star:>8.2e}\t{err:>8.2e}\t{err_pct:>7.2%}\t{tokens_per_param:>8.2f}\t{d:>4}\t{L:>3}\t{aspect_ratio:>5.2f}\t{h:>3}\t{lr:>9.2e}\t{embed_ratio:>9.2f}\t{tokens:>9.2e}\t{tokens_per_non_embedding:>8.2f}"
        )

    print("-" * 100)


if __name__ == "__main__":
    print_predicted_shapes(n_for_c_fn=lambda c: int(get_chinchilla_n_for_c(c)))

    n = int(get_chinchilla_n_for_c(1e15) * 0.5)
    candidates = pick_candidates_around_n(
        n, factor=3, n_candidates=7, round_to_int=True
    )
    # shapes = [get_shape_given_n(n) for n in candidates]
    print_predicted_shapes(ns=candidates)

    # print(get_chinchilla_lr_for_n(n))
