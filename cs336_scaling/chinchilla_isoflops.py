"""
This script fits a power law to empirical data and extrapolates the optimal parameter
counts for larger compute budgets.

The empirical data is stored in `data/isoflops_curves.json`, which is a list of
dictionaries in which each dictionary represents a training run with a particular
compute budget in FLOPs and a transformer with the specified number or parameters,
and reports the final loss of the model at the end of the training run.

Each dictionary in the list of runs has the following keys:
- `compute_budget`: the compute budget in FLOPs
- `parameters`: the number of parameters in the model
- `final_loss`: the final loss of the model

The script prints the empirical data, the fitted power law, and the extrapolated data.

The script also plots the empirical data, the fitted power law, and the extrapolated data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import defaultdict
import os
import logging
import sys
from cs336_scaling.common import power_law


def load_and_process_data(filepath: str) -> tuple[list[float], dict[float, int]]:
    """Loads data from JSON, groups by budget, and finds optimal parameters."""
    with open(filepath, "r") as f:
        data = json.load(f)

    data_by_budget = defaultdict(list)
    for d in data:
        # Ensure compute_budget is float for consistent key types
        data_by_budget[float(d["compute_budget"])].append(d)

    budgets = sorted(data_by_budget.keys())

    c_to_n_opt = {}
    for budget in budgets:
        budget_data = data_by_budget[budget]
        min_loss_entry = min(budget_data, key=lambda x: x["final_loss"])
        c_to_n_opt[budget] = min_loss_entry["parameters"]

    return budgets, c_to_n_opt


def format_with_suffix(value: float) -> str:
    """Formats a number with magnitude suffix (T, B, M, K)."""
    suffixes = [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]
    for suffix, threshold in suffixes:
        if abs(value) >= threshold:
            return f"{int(value):,} ({value / threshold:.2f}{suffix}) ({value:.2e})"

    return f"{int(value):,} ({value:.2e})"


def print_results(c_to_n_opt: dict[float, int]):
    """Prints compute budget, optimal params (N), and optimal tokens (D)."""
    logging.info(
        f"{'Compute (C)':<15}{'Optimal Params (N)':<40}{'Optimal Tokens (D)':<40}"
    )
    logging.info("-" * 95)

    for budget in sorted(c_to_n_opt.keys()):
        n_opt = c_to_n_opt[budget]
        d_opt = budget / (6 * n_opt)

        logging.info(
            f"{budget:<15.2e}"
            f"{format_with_suffix(n_opt):<40}"
            f"{format_with_suffix(d_opt):<40}"
        )
    logging.info("\n")


def fit_power_law(c_to_n_opt: dict[float, int]) -> tuple[float, float]:
    """Fits a power law N = a * C^b to the optimal parameter data."""
    budgets = sorted(list(c_to_n_opt.keys()))
    x = np.array(budgets)
    y = np.array([c_to_n_opt[b] for b in budgets])
    popt, _ = curve_fit(
        power_law, x, y, p0=(25, 0.5)
    )  # Set initial guess for exponent = 0.5 based on Hoffman et. al
    return tuple(popt)


def extrapolate_optimal_params(
    popt: tuple[float, float], min_compute: float = 1e20, max_compute: float = 1e25
) -> dict[float, float]:
    """Extrapolates optimal parameters using the fitted power law."""
    a, b = popt
    extended_opts = {}
    min_oom = int(np.floor(np.log10(min_compute)))
    max_oom = int(np.ceil(np.log10(max_compute)))

    # Generate compute points: 1eX, 3eX, 6eX within the range
    compute_points = []
    for oom in range(min_oom, max_oom):
        for multiplier in [1, 3, 6]:
            compute = multiplier * (10**oom)
            if min_compute <= compute <= max_compute:
                compute_points.append(compute)

    if max_compute not in compute_points and max_compute >= min_compute:
        if not compute_points or max_compute > compute_points[-1]:
            compute_points.append(max_compute)
    compute_points.sort()

    for compute in compute_points:
        extended_opts[compute] = power_law(np.array([compute]), a, b)[0]

    return extended_opts


def plot_params_vs_compute(
    budgets: list[float],
    c_to_n_opt: dict[float, int],
    popt: tuple[float, float],
    c_to_n_opt_extended: dict[float, float],
    save_path: str,
):
    """Plots empirical and extrapolated optimal N vs C and the power law fit."""
    a, b = popt
    x_empirical = np.array(budgets)
    y_empirical = np.array([c_to_n_opt[b] for b in budgets])

    # Sort extended data by compute budget for consistent plotting
    extended_items = sorted(c_to_n_opt_extended.items())
    x_extended = np.array([item[0] for item in extended_items])
    y_extended = np.array([item[1] for item in extended_items])

    plt.figure(figsize=(10, 6))
    plt.scatter(
        x_empirical, y_empirical, label="Empirically Optimal N", marker="o", s=50
    )
    plt.scatter(
        x_extended,
        y_extended,
        label="Extrapolated Optimal N",
        marker="x",
        s=50,
        c="orange",
    )

    # Determine plot range
    min_x = x_empirical.min()  # Extrapolation starts after empirical data
    max_x = x_extended.max()

    x_line = np.geomspace(min_x, max_x, 500)
    y_line = power_law(x_line, a, b)
    plt.plot(
        x_line,
        y_line,
        label=f"Fit: N = {a:.2e} * C^{b:.2f}",
        color="red",
        linestyle="--",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Compute Budget (C) (FLOPs)")
    plt.ylabel("Optimal Parameters (N)")
    plt.title("Optimal Parameters vs Compute Budget (Chinchilla Scaling)")
    plt.legend()
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_data_vs_compute(
    budgets: list[float],
    c_to_n_opt: dict[float, int],
    popt: tuple[float, float],
    c_to_n_opt_extended: dict[float, float],
    save_path: str,
):
    """Plots empirical and extrapolated optimal D vs C and the derived power law fit."""
    a_n, b_n = popt  # N = a_n * C^b_n

    # Derived power law for D = C / (6N) -> D = (1 / (6*a_n)) * C^(1-b_n)
    a_d = 1 / (6 * a_n)
    b_d = 1 - b_n

    x_empirical = np.array(budgets)
    y_empirical_n = np.array([c_to_n_opt[b] for b in budgets])
    y_empirical_d = x_empirical / (6 * y_empirical_n)  # D = C / 6N

    # Sort extended data by compute budget
    extended_items = sorted(c_to_n_opt_extended.items())
    x_extended = np.array([item[0] for item in extended_items])
    y_extended_n = np.array([item[1] for item in extended_items])
    y_extended_d = x_extended / (6 * y_extended_n)  # D = C / 6N

    plt.figure(figsize=(10, 6))
    plt.scatter(
        x_empirical, y_empirical_d, label="Empirically Optimal D", marker="o", s=50
    )
    plt.scatter(
        x_extended,
        y_extended_d,
        label="Extrapolated Optimal D",
        marker="x",
        s=50,
        c="orange",
    )

    # Determine plot range
    min_x = x_empirical.min()
    max_x = x_extended.max()

    # Plot derived power law fit for D
    x_line = np.geomspace(min_x, max_x, 500)
    y_line = power_law(x_line, a_d, b_d)
    plt.plot(
        x_line,
        y_line,
        label=f"Derived Fit: D = {a_d:.2e} * C^{b_d:.2f}",
        color="red",
        linestyle="--",
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Compute Budget (C) (FLOPs)")
    plt.ylabel("Optimal Dataset Size (D) (Tokens)")
    plt.title("Optimal Dataset Size vs Compute Budget (Chinchilla Scaling)")
    plt.legend()
    plt.grid(True, which="major", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def fit_hoffman_data():
    flops_to_params = {
        1.92e19: 400_000_000,
        1.21e20: 1_000_000_000,
        1.23e22: 10_000_000_000,
        5.76e23: 67_000_000_000,
        3.85e24: 175_000_000_000,
        9.90e24: 280_000_000_000,
        3.43e25: 520_000_000_000,
        1.27e26: 1_000_000_000_000,
        1.30e28: 10_000_000_000_000,
    }

    popt = fit_power_law(flops_to_params)
    a, b = popt
    print(f"Fitted power law: N = {a:.6e} * C^{b:.6f}\n")


def main():
    """Main execution function."""
    data_filepath = "data/isoflops_curves.json"
    out_dir = "out/chinchilla"
    params_plot_path = os.path.join(out_dir, "params-current.png")
    data_plot_path = os.path.join(out_dir, "data-current.png")
    log_file_path = os.path.join(out_dir, "log-current.txt")

    os.makedirs(out_dir, exist_ok=True)

    # Configure logging
    log_formatter = logging.Formatter("%(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(
        log_file_path, mode="w"
    )  # Overwrite log file each run
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    budgets, c_to_n_opt = load_and_process_data(data_filepath)

    logging.info("--- Empirical Data ---")
    print_results(c_to_n_opt)

    popt = fit_power_law(c_to_n_opt)
    a, b = popt
    logging.info(f"Fitted power law: N = {a:.6e} * C^{b:.6f}\n")

    # Extrapolate, starting just above last empirical budget
    extrapolation_start_compute = sorted(budgets)[-1] * (1 + 1e-10)
    max_compute_budget = 1e25
    c_to_n_opt_extended = extrapolate_optimal_params(
        popt, extrapolation_start_compute, max_compute_budget
    )

    logging.info("--- Extrapolated Data ---")
    print_results(c_to_n_opt_extended)

    # Plot N vs C and save
    plot_params_vs_compute(
        budgets, c_to_n_opt, popt, c_to_n_opt_extended, params_plot_path
    )
    logging.info(f"Saved parameter plot to {params_plot_path}")

    # Plot D vs C and save
    plot_data_vs_compute(budgets, c_to_n_opt, popt, c_to_n_opt_extended, data_plot_path)
    logging.info(f"Saved data size plot to {data_plot_path}")


if __name__ == "__main__":
    main()
    # fit_hoffman_data()
