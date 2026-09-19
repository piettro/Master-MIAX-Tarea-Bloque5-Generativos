"""Command-line entry point for the synthetic financial data pipeline.

Usage examples:
    python main.py build-dataset
    python main.py search-architecture
    python main.py train-baseline
    python main.py train-generator cgan
    python main.py evaluate-generator cgan
    python main.py threshold-sensitivity
    python main.py compare
    python main.py run-all --skip-diffusion

Run ``python main.py --help`` or ``python main.py <command> --help`` for the
full list of commands and their options.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from src.utils.config import (
    MODEL_DISPLAY_NAMES,
    SETTINGS,
    ensure_directories,
)
from src.utils.logging_config import configure_logging

# Every subcommand below imports Keras-dependent modules lazily, inside its
# own function body, specifically so this line runs first and the PyTorch
# backend is selected before Keras itself is ever imported.
os.environ.setdefault("KERAS_BACKEND", "torch")

logger = logging.getLogger(__name__)

GENERATOR_NAMES: tuple[str, ...] = ("noise", "cgan", "cvae", "diffusion")


def _load_dataset_and_budget():
    """Load the processed dataset and the fixed real-data training budget."""
    from src.data.dataset import load_dataset, sample_real_budget

    data = load_dataset()
    x_real, y_real = sample_real_budget(
        data["X_train"], data["y_train"], SETTINGS.n_real_samples, SETTINGS.seed
    )
    return data, x_real, y_real


def cmd_download_prices(_: argparse.Namespace) -> None:
    """Download the full S&P 500 price history and refresh the cache."""
    from src.data.loader import download_price_universe

    download_price_universe()


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Build the processed dataset from the cached price universe."""
    import numpy as np

    from src.data.dataset import (
        build_windows,
        calibrate_threshold,
        clean_returns,
        compute_log_returns,
        future_drawdown,
        label_windows,
        market_series,
        save_dataset,
        temporal_split,
    )
    from src.data.dataset import TanhScaler
    from src.data.loader import load_price_universe

    ensure_directories()

    prices = load_price_universe()
    logger.info(
        "Universe: %d assets, %s to %s", prices.shape[1], prices.index.min().date(),
        prices.index.max().date(),
    )

    raw_returns = compute_log_returns(prices)
    returns, n_clipped = clean_returns(raw_returns)
    logger.info("Clipped %d implausible return values.", n_clipped)

    drawdown = future_drawdown(market_series(returns))
    x, drawdowns, dates = build_windows(returns, drawdown)
    logger.info("Built %d windows of shape %s.", len(x), x.shape[1:])

    train_idx, val_idx, test_idx = temporal_split(len(x))
    threshold = calibrate_threshold(drawdowns[train_idx])
    y = label_windows(drawdowns, threshold)
    logger.info(
        "Crisis threshold: %.2f%%. Positive rate -- train %.1f%%, val %.1f%%, "
        "test %.1f%%.",
        100 * threshold,
        100 * y[train_idx].mean(),
        100 * y[val_idx].mean(),
        100 * y[test_idx].mean(),
    )

    scaler = TanhScaler().fit(x[train_idx])
    x_scaled = scaler.transform(x)

    save_dataset(
        SETTINGS.dataset_path,
        X_train=x_scaled[train_idx],
        y_train=y[train_idx],
        X_val=x_scaled[val_idx],
        y_val=y[val_idx],
        X_test=x_scaled[test_idx],
        y_test=y[test_idx],
        dd_train=drawdowns[train_idx],
        dd_val=drawdowns[val_idx],
        dd_test=drawdowns[test_idx],
        dates_train=dates[train_idx].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        dates_val=dates[val_idx].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        dates_test=dates[test_idx].strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        tickers=np.array(prices.columns, dtype=object),
        drawdown_threshold=np.array(threshold),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        scaler_n_sigmas=np.array(scaler.n_sigmas),
    )
    logger.info("Dataset saved to %s", SETTINGS.dataset_path)


def cmd_threshold_sensitivity(_: argparse.Namespace) -> None:
    """Sweep the crisis-labelling threshold and report episode counts.

    NOTE: added -- required by the professor's feedback (see
    src.evaluation.experiment.threshold_sensitivity for the full rationale).
    """
    from src.data.dataset import clean_returns, compute_log_returns
    from src.data.loader import load_price_universe
    from src.evaluation.experiment import threshold_sensitivity
    from src.utils.config import ensure_directories

    ensure_directories()
    prices = load_price_universe()
    returns, _ = clean_returns(compute_log_returns(prices))

    table = threshold_sensitivity(returns)
    path = SETTINGS.tables_dir / "threshold_sensitivity.csv"
    table.to_csv(path, index=False)

    logger.info("Threshold sensitivity saved to %s", path)
    for _, row in table.iterrows():
        logger.info(
            "positive_rate=%.3f -> threshold=%.2f%%  positive_windows=%d  episodes=%d",
            row["positive_rate"],
            100 * row["drawdown_threshold"],
            row["n_positive_windows"],
            row["n_episodes"],
        )


def cmd_search_architecture(args: argparse.Namespace) -> None:
    """Run the architecture search on the real-data budget."""
    from src.models.classifier import search_architecture

    _, x_real, y_real = _load_dataset_and_budget()
    from src.data.dataset import load_dataset

    data = load_dataset()

    table = search_architecture(
        x_real, y_real, data["X_val"], data["y_val"], n_seeds=args.seeds
    )
    path = SETTINGS.tables_dir / "architecture_search.csv"
    ensure_directories()
    table.to_csv(path, index=False)
    logger.info("Architecture search results saved to %s", path)


def cmd_train_baseline(_: argparse.Namespace) -> None:
    """Evaluate the classifier on the real-data budget alone (ratio = 0)."""
    from src.evaluation.experiment import ratio_sweep, save_results, summarize

    data, x_real, y_real = _load_dataset_and_budget()
    results, _ = ratio_sweep(
        "baseline",
        x_real,
        y_real,
        None,
        None,
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        ratios=[0.0],
    )
    path = save_results(results, "baseline")
    logger.info("Baseline results saved to %s", path)
    logger.info("\n%s", summarize(results).round(4).to_string(index=False))


def cmd_train_generator(args: argparse.Namespace) -> None:
    """Train one generator and persist its synthetic dataset and loss curves."""
    from src.evaluation.experiment import save_synthetic_data
    from src.models.generators import create_generator
    from src.visualization import plots

    plots.apply_style()
    _, x_real, y_real = _load_dataset_and_budget()

    generator = create_generator(args.generator, seed=SETTINGS.seed)
    logger.info("Training generator '%s'...", args.generator)
    generator.fit(x_real, y_real)

    if args.generator == "diffusion":
        generator.calibrate_temperature(x_real, y_real)

    n_synthetic = int(SETTINGS.n_real_samples * max(SETTINGS.synthetic_ratios))
    x_synthetic, y_synthetic = generator.generate(n_synthetic, float(y_real.mean()))

    save_synthetic_data(args.generator, x_synthetic, y_synthetic, generator.loss_history)

    for curve_name, values in generator.loss_history.items():
        if values:
            plots.loss_curve(
                {"loss": values},
                f"{MODEL_DISPLAY_NAMES.get(args.generator, args.generator)} -- {curve_name}",
                f"{args.generator}_loss_{curve_name}",
                keys=("loss",),
            )

    logger.info(
        "Generated %d synthetic windows for '%s'. Positive rate: %.1f%%.",
        len(x_synthetic),
        args.generator,
        100 * float(y_synthetic.mean()),
    )


def cmd_evaluate_generator(args: argparse.Namespace) -> None:
    """Run the ratio sweep for one generator's saved synthetic dataset."""
    from src.evaluation.experiment import (
        load_synthetic_data,
        ratio_sweep,
        save_results,
        summarize,
    )
    from src.visualization import plots

    plots.apply_style()
    data, x_real, y_real = _load_dataset_and_budget()
    synthetic = load_synthetic_data(args.generator)

    results, histories = ratio_sweep(
        args.generator,
        x_real,
        y_real,
        synthetic["x_synthetic"],
        synthetic["y_synthetic"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )
    path = save_results(results, args.generator)
    logger.info("Results for '%s' saved to %s", args.generator, path)

    summary = summarize(results)
    logger.info("\n%s", summary.round(4).to_string(index=False))

    plots.ratio_curve(
        summary,
        title=f"{MODEL_DISPLAY_NAMES.get(args.generator, args.generator)}: effect on test",
        file_name=f"{args.generator}_ratio_curve",
    )


def cmd_compare(_: argparse.Namespace) -> None:
    """Build the final cross-generator comparison, statistics, and figures."""
    from src.evaluation.experiment import load_results, load_synthetic_data, summarize
    from src.evaluation.mutual_information import mutual_information_table
    from src.evaluation.statistics import (
        build_comparison_table,
        paired_significance_test,
        save_comparison_table,
        save_significance_table,
        summarize_significant_effects,
    )
    from src.visualization import plots

    plots.apply_style()
    model_names = ["baseline", *GENERATOR_NAMES]
    results = load_results(model_names)
    summary = summarize(results)

    plots.ratio_curve(
        summary,
        metric="pr_auc_mean",
        std_column="pr_auc_std",
        title="Effect of synthetic data on the test set",
        file_name="comparison_test",
    )
    plots.ratio_curve(
        summary,
        metric="pr_auc_val_mean",
        std_column="pr_auc_val_std",
        title="Effect of synthetic data on validation",
        file_name="comparison_validation",
    )

    contrasts = paired_significance_test(results)
    save_significance_table(contrasts)
    for line in summarize_significant_effects(contrasts):
        logger.info(line)
    if contrasts.empty or not (
        (contrasts["p_test"] < 0.05) | (contrasts["p_val"] < 0.05)
    ).any():
        logger.info("No difference reaches significance at the 5%% level.")

    comparison = build_comparison_table(summary, contrasts, MODEL_DISPLAY_NAMES)
    save_comparison_table(comparison)
    logger.info("\n%s", comparison.round(4).to_string(index=False))

    _, x_real, y_real = _load_dataset_and_budget()
    mi_datasets = [("Real", x_real, y_real)]
    for name in GENERATOR_NAMES:
        try:
            synthetic = load_synthetic_data(name)
            mi_datasets.append(
                (
                    MODEL_DISPLAY_NAMES.get(name, name),
                    synthetic["x_synthetic"],
                    synthetic["y_synthetic"],
                )
            )
        except FileNotFoundError:
            logger.warning("No synthetic data for '%s'; skipped in mutual information.", name)

    mi_table = mutual_information_table(mi_datasets, len(x_real), SETTINGS.seed)
    ensure_directories()
    mi_table.to_csv(SETTINGS.tables_dir / "mutual_information.csv")
    logger.info("\n%s", mi_table.round(4).to_string())


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run the full pipeline end to end.

    This trains every generator from scratch, which is expensive (the
    diffusion model alone takes on the order of 20-35 minutes on a
    conventional CPU); see README.md for measured timings of each stage.
    """
    cmd_build_dataset(args)
    cmd_train_baseline(args)

    generators = [g for g in GENERATOR_NAMES if not (g == "diffusion" and args.skip_diffusion)]
    for name in generators:
        train_args = argparse.Namespace(generator=name)
        cmd_train_generator(train_args)
        cmd_evaluate_generator(train_args)

    cmd_compare(args)


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic financial data pipeline: can adding synthetic windows "
            "improve a classifier that predicts severe S&P 500 drawdowns?"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "download-prices", help="Download the full S&P 500 price history."
    ).set_defaults(func=cmd_download_prices)

    subparsers.add_parser(
        "build-dataset", help="Build the processed dataset from the cached prices."
    ).set_defaults(func=cmd_build_dataset)

    subparsers.add_parser(
        "threshold-sensitivity",
        help="Sweep the crisis-labelling threshold and report episode counts.",
    ).set_defaults(func=cmd_threshold_sensitivity)

    search_parser = subparsers.add_parser(
        "search-architecture", help="Search the classifier architecture grid."
    )
    search_parser.add_argument("--seeds", type=int, default=3)
    search_parser.set_defaults(func=cmd_search_architecture)

    subparsers.add_parser(
        "train-baseline", help="Evaluate the classifier with real data only."
    ).set_defaults(func=cmd_train_baseline)

    train_gen_parser = subparsers.add_parser(
        "train-generator", help="Train one generator and save its synthetic data."
    )
    train_gen_parser.add_argument("generator", choices=GENERATOR_NAMES)
    train_gen_parser.set_defaults(func=cmd_train_generator)

    eval_gen_parser = subparsers.add_parser(
        "evaluate-generator", help="Run the ratio sweep for one trained generator."
    )
    eval_gen_parser.add_argument("generator", choices=GENERATOR_NAMES)
    eval_gen_parser.set_defaults(func=cmd_evaluate_generator)

    subparsers.add_parser(
        "compare", help="Build the final comparison across every generator."
    ).set_defaults(func=cmd_compare)

    run_all_parser = subparsers.add_parser(
        "run-all", help="Run the full pipeline end to end."
    )
    run_all_parser.add_argument(
        "--skip-diffusion",
        action="store_true",
        help="Skip the diffusion generator, the slowest stage (~20-35 min).",
    )
    run_all_parser.add_argument("--seeds", type=int, default=3)
    run_all_parser.set_defaults(func=cmd_run_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults
            to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` on a handled error.
    """
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except Exception:
        logger.exception("Unhandled error while running '%s'.", args.command)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
