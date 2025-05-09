import json
import matplotlib.pyplot as plt
import numpy as np
from cs336_scaling.common import power_law
from cs336_scaling.training_api import sync_api_state
from scipy.optimize import curve_fit


def get_all_runs(sync_api: bool = False):
    if sync_api:
        sync_api_state()

    with open("out/runs.json", "r") as f:
        return list(json.load(f).values())


def get_runs_after(start_time: float, sync_api: bool = False):
    runs = get_all_runs(sync_api=sync_api)
    return [run for run in runs if run["sync_time"] > start_time]


def best_run_where(key: str, value: float, runs: list[dict] | None = None):
    if runs is None:
        runs = get_all_runs()

    matching_runs = [run for run in runs if run[key] == value]
    if not matching_runs:
        return None
    return min(matching_runs, key=lambda x: x["loss"])


def group_by(items: list[dict], key: str) -> dict[str, list[dict]]:
    grouped = {}
    for item in items:
        if item[key] not in grouped:
            grouped[item[key]] = []
        grouped[item[key]].append(item)
    return grouped


def fit_quadratic(runs: list[dict]) -> np.poly1d:
    """
    Fits a quadratic function to the data.

    Args:
        runs: A list of dictionaries containing run data.

    Returns:
        A numpy poly1d object representing the fit
        loss = f(log(params)).
    """
    non_embedding_params = np.array([run["est_n_non_embedding"] for run in runs])
    losses = np.array([run["loss"] for run in runs])
    log_params = np.log(non_embedding_params)
    coeffs = np.polyfit(log_params, losses, 2)
    func = np.poly1d(coeffs)
    return func


def fit_power_law(c_to_n_opt: dict[float, int]) -> tuple[float, float]:
    """Fits a power law N = a * C^b to the optimal parameter data."""
    budgets = sorted(list(c_to_n_opt.keys()))
    x = np.array(budgets)
    y = np.array([c_to_n_opt[b] for b in budgets])

    # Set initial guess for exponent = 0.5 based on Hoffman et. al
    popt, _ = curve_fit(power_law, x, y, p0=(25, 0.5))
    return tuple(popt)


def find_optimal_params(quadratic_fit: np.poly1d) -> float:
    """
    Finds the parameter count that minimizes the loss for a fitted quadratic.

    Args:
        quadratic_fit: A numpy poly1d object representing the fit
                       loss = f(log(params)).

    Returns:
        The estimated optimal number of non-embedding parameters.
    """

    coeffs = quadratic_fit.coeffs

    # Check if the parabola opens upwards (a > 0) for a minimum
    if coeffs[0] <= 0:
        print(
            "Warning: Fitted quadratic does not have a minimum (opens downwards or is linear)."
        )
        return np.nan

    log_params_min = -coeffs[1] / (2 * coeffs[0])
    params_min = np.exp(log_params_min)
    return params_min


def plot_loss_vs_lr_at_c_and_n(
    c: float, n: float, show_plot: bool = False, save_plot: bool = True
):
    """
    Plot the loss vs. learning rate for a given training FLOPs and non-embedding parameters.

    Args:
        - c: The training FLOPs to plot.
        - n: The non-embedding parameters to plot.
        - show_plot: Whether to show the plot.
        - save_plot: Whether to save the plot.
    """
    runs = get_all_runs()
    runs_at_c_and_n = [
        run
        for run in runs
        if run["train_flops"] == c and run["est_n_non_embedding"] == n
    ]

    lrs = [run["learning_rate"] for run in runs_at_c_and_n]
    losses = [run["loss"] for run in runs_at_c_and_n]

    plt.figure(figsize=(10, 6))
    plt.scatter(lrs, losses, alpha=0.8)
    plt.xscale("log")
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Loss")
    plt.title(
        f"Loss vs. Learning Rate at {c:.0e} FLOPs and {n:.0e} non-embedding parameters"
    )
    plt.tight_layout()

    if save_plot:
        plt.savefig(f"out/loss_vs_lr_at_{c:.0e}_and_{n:.0e}.png")

    if show_plot:
        plt.show()


def plot_runs(
    runs: list[dict] | None = None,
    best_n_per_c: int | None = None,
    show_plot: bool = False,
    save_plot: bool = True,
    outfile_suffix: str | None = None,
    quadratics: bool = True,
    use_best_lr_per_n: bool = True,
    sync_api: bool = False,
):
    """
    Plot the runs in a scatter plot

    - One group of points for each unique value of `train_flops`
    - x-axis: Non-embedding parameters (log scale)
    - y-axis: Loss
    - Color: `train_flops`
    """
    if sync_api:
        sync_api_state()

    if runs is None:
        runs = get_all_runs()

    runs_by_train_flops = group_by(runs, "train_flops")

    if best_n_per_c is not None:
        for tf in runs_by_train_flops:
            group = runs_by_train_flops[tf]

            if use_best_lr_per_n:
                group = best_lr_per_n(group)
                group_by_n = group_by(group, "est_n_non_embedding")
                for n in group_by_n:
                    group_by_n[n] = sorted(group_by_n[n], key=lambda x: x["loss"])[0]

                group = list(group_by_n.values())

            runs_by_train_flops[tf] = sorted(group, key=lambda x: x["loss"])[
                :best_n_per_c
            ]

    # Prepare for plotting
    train_flops_values = sorted(runs_by_train_flops.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(train_flops_values)))

    plt.figure(figsize=(10, 6))

    for i, tf in enumerate(train_flops_values):
        group = runs_by_train_flops[tf]
        non_embedding_params = np.array([run["est_n_non_embedding"] for run in group])
        losses = np.array([run["loss"] for run in group])

        # Format train_flops label in scientific notation
        label = f"{tf:.0e} FLOPs"

        plt.scatter(
            non_embedding_params, losses, color=colors[i], label=label, alpha=0.8
        )

        # Fit and plot quadratic if enough points exist
        if len(non_embedding_params) >= 3 and quadratics:
            log_params = np.log(non_embedding_params)
            coeffs = np.polyfit(log_params, losses, 2)
            fit_fn = np.poly1d(coeffs)

            # Generate points for the fitted curve
            log_params_plot = np.linspace(log_params.min(), log_params.max(), 100)
            losses_plot = fit_fn(log_params_plot)
            params_plot = np.exp(log_params_plot)

            plt.plot(params_plot, losses_plot, color=colors[i], linestyle="dotted")

    plt.xscale("log")
    plt.xlabel("Non-Embedding Parameters (Log Scale)")
    plt.ylabel("Loss")
    plt.title("Loss vs. Non-Embedding Parameters by Training FLOPs")
    plt.legend(title="Train FLOPs")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    if save_plot:
        plt.savefig(
            f"out/ne_params_vs_loss{outfile_suffix if outfile_suffix else ''}.png"
        )

    if show_plot:
        plt.show()


def best_lr_per_n(runs: list[dict]) -> list[dict]:
    runs_by_n = group_by(runs, "est_n_non_embedding")
    for n in runs_by_n:
        runs_by_n[n] = sorted(runs_by_n[n], key=lambda x: x["loss"])[0]
    return list(runs_by_n.values())


def plot_tokens_per_param_vs_loss(
    runs: list[dict] | None = None,
    best_n_per_c: int | None = None,
    show_plot: bool = False,
    save_plot: bool = True,
    outfile_suffix: str | None = None,
    use_best_lr_per_n: bool = True,
    sync_api: bool = False,
):
    if sync_api:
        sync_api_state()

    if runs is None:
        runs = get_all_runs()

    runs_by_train_flops = group_by(runs, "train_flops")

    if best_n_per_c is not None:
        for tf in runs_by_train_flops:
            group = runs_by_train_flops[tf]

            if use_best_lr_per_n:
                group = best_lr_per_n(group)

            runs_by_train_flops[tf] = sorted(group, key=lambda x: x["loss"])[
                :best_n_per_c
            ]

    # Prepare for plotting
    train_flops_values = sorted(runs_by_train_flops.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(train_flops_values)))

    plt.figure(figsize=(10, 6))

    for i, tf in enumerate(train_flops_values):
        group = runs_by_train_flops[tf]
        tokens_per_param = np.array([run["est_tokens_per_param"] for run in group])
        losses = np.array([run["loss"] for run in group])

        label = f"{tf:.0e} FLOPs"
        plt.scatter(tokens_per_param, losses, color=colors[i], label=label, alpha=0.8)

    plt.xlabel("Tokens per Parameter")
    plt.ylabel("Loss")
    plt.title("Loss vs. Tokens per Parameter by Training FLOPs")
    plt.legend(title="Train FLOPs")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    if save_plot:
        plt.savefig(
            f"out/tokens_per_param_vs_loss{outfile_suffix if outfile_suffix else ''}.png"
        )

    if show_plot:
        plt.show()


if __name__ == "__main__":  # print(best_run_where("train_flops", 1e14))
    C = 1e16
    all_runs = get_all_runs()
    runs_at_c = [run for run in all_runs if run["train_flops"] == C]
    plot_runs(runs=runs_at_c, best_n_per_c=5, outfile_suffix=f"_{C}.png")

    # best_run_at_1e14 = best_run_where("train_flops", 1e14)
    # plot_loss_vs_lr_at_c_and_n(
    #     1e14,
    #     best_run_at_1e14["est_n_non_embedding"],
    #     show_plot=True,
    #     save_plot=True,
    # )
