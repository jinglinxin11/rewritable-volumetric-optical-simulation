"""Validate archived source tables and regenerate Supplementary Figs. 12/13.

This reporting layer does not train, select checkpoints, or modify source data.
The same official test images are reused across runs, not independent cohorts.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np

SELECTED, REFERENCE = "c12_k5_s1", "c3_k3_s1"
SEEDS = (3, 4, 5)
SOURCE_TABLES = frozenset({
    'class_counts.csv', 'confirmation_metrics.csv', 'depth_encoding.csv',
    'mean_confusion_percent.csv', 'resources.csv', 'search_run_metrics.csv',
    'selected_training_history.csv', 'test_predictions.csv', 'validation_ranking.csv',
})


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Labels and predictions must be nonempty matching vectors")
    for values in (labels, predictions):
        if not np.issubdtype(values.dtype, np.integer) or np.any((values < 0) | (values > 9)):
            raise ValueError("MNIST classes must be integers in [0, 9]")
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    denominator = matrix.sum(0) + matrix.sum(1)
    # Empty classes contribute zero; every class is present in official MNIST.
    f1_percent = np.divide(200.0 * np.diag(matrix), denominator,
                           out=np.zeros(10), where=denominator > 0)
    return {"accuracy_percent": float(100 * np.mean(labels == predictions)),
            "macro_f1_percent": float(f1_percent.mean()), "confusion_counts": matrix}


def load_tables(source, required_files=()):
    source = Path(source)
    if source.is_dir():
        members = {p.relative_to(source).as_posix(): p.read_bytes()
                   for p in source.rglob("*") if p.is_file()}
    else:
        with ZipFile(source) as archive:
            names = [n for n in archive.namelist() if not n.endswith('/')]
            if len(names) != len(set(names)):
                raise ValueError('Duplicate archive member')
            members = {n: archive.read(n) for n in names}
    manifest = json.loads(members["manifest.json"])
    entries = manifest if isinstance(manifest, list) else manifest["files"]
    paths = [row['file'] for row in entries]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate manifest path')
    if any(not isinstance(name, str) or PurePosixPath(name).is_absolute()
           or '..' in PurePosixPath(name).parts or '\\' in name for name in paths):
        raise ValueError('Invalid manifest path')
    actual = set(members) - {'manifest.json'}
    if set(paths) != actual:
        raise ValueError(f'Manifest coverage mismatch: unlisted={sorted(actual-set(paths))}, '
                         f'missing={sorted(set(paths)-actual)}')
    if not set(required_files).issubset(actual):
        raise ValueError(f'Missing required source files: {sorted(set(required_files)-actual)}')
    for row in entries:
        if hashlib.sha256(members[row["file"]]).hexdigest() != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {row['file']}")
        if 'bytes' in row and row['bytes'] != len(members[row['file']]):
            raise ValueError(f"Size mismatch: {row['file']}")
    tables = {name: list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
              for name, data in members.items() if name.endswith(".csv")}
    return tables, members


def confirmation_records(rows):
    keys = [(row['architecture'], int(row['seed'])) for row in rows]
    expected = {(architecture, seed) for architecture in (REFERENCE, SELECTED) for seed in SEEDS}
    if len(keys) != 6 or len(set(keys)) != 6 or set(keys) != expected:
        raise ValueError('Expected exactly six unique confirmation records for seeds 3, 4, 5')
    return dict(zip(keys, rows))


def validate_tables(tables):
    if not SOURCE_TABLES.issubset(tables):
        raise ValueError(f'Missing required source tables: {sorted(SOURCE_TABLES-set(tables))}')
    metrics = confirmation_records(tables['confirmation_metrics.csv'])
    records = tables["test_predictions.csv"]
    if [int(r["official_test_index"]) for r in records] != list(range(10000)):
        raise ValueError("Expected exactly the 10,000 ordered official test indices")
    labels = np.array([int(r["true_digit"]) for r in records])
    results, matrices, summary = [], [], {}
    for architecture in (REFERENCE, SELECTED):
        rows = []
        for seed in SEEDS:
            prediction = np.array([int(r[f"{architecture}_seed{seed}"]) for r in records])
            computed = classification_metrics(labels, prediction)
            reported = metrics[architecture, seed]
            for field, key in (("test_accuracy_percent", "accuracy_percent"),
                               ("macro_f1_percent", "macro_f1_percent")):
                if not np.isclose(float(reported[field]), computed[key], rtol=0, atol=1e-10):
                    raise ValueError(f"Metric mismatch: {architecture}, seed {seed}, {field}")
            row = {"architecture": architecture, "seed": seed,
                   "selected_epoch": int(reported["selected_epoch"]),
                   "accuracy_percent": computed["accuracy_percent"],
                   "macro_f1_percent": computed["macro_f1_percent"]}
            rows.append(row)
            if architecture == SELECTED:
                cm = computed["confusion_counts"]
                matrices.append(100.0 * cm / cm.sum(1, keepdims=True))
        results.extend(rows)
        summary[architecture] = {
            key: {"mean": float(np.mean([r[key] for r in rows])),
                  "population_sd_pp": float(np.std([r[key] for r in rows], ddof=0))}
            for key in ("accuracy_percent", "macro_f1_percent")}
    mean_matrix = np.mean(matrices, axis=0)
    supplied = np.full((10, 10), np.nan)
    for row in tables["mean_confusion_percent.csv"]:
        i, j = int(row["true_digit"]), int(row["predicted_digit"])
        if not np.isnan(supplied[i, j]):
            raise ValueError("Duplicate confusion entry")
        supplied[i, j] = float(row["percent"])
    np.testing.assert_allclose(supplied, mean_matrix, rtol=0, atol=1e-10)
    np.testing.assert_allclose(mean_matrix.sum(1), 100, rtol=0, atol=1e-10)
    if (len(tables['class_counts.csv']) != 10
            or {int(row['digit']) for row in tables['class_counts.csv']} != set(range(10))):
        raise ValueError('Expected ten unique class-count records')
    for row in tables["class_counts.csv"]:
        if int(row["test"]) != int(np.sum(labels == int(row["digit"]))):
            raise ValueError("Test class counts differ from predictions")
    for seed in SEEDS:
        history = [r for r in tables["selected_training_history.csv"] if int(r["seed"]) == seed]
        if [int(r["epoch"]) for r in history] != list(range(21)) or history[0]["loss"]:
            raise ValueError("Expected epoch 0 without loss and epochs 1 through 20")
        best = max(float(r["validation_percent"]) for r in history)
        epoch = next(int(r["epoch"]) for r in history if float(r["validation_percent"]) == best)
        if epoch != int(metrics[SELECTED, seed]["selected_epoch"]):
            raise ValueError("Checkpoint is not earliest maximum validation epoch")
        if int(history[-1]["virtual_erasures"]) != 1961:
            raise ValueError("Unexpected complete-training reset count")
    ranking = tables["validation_ranking.csv"]
    if len(ranking) != 12 or ranking[0]["architecture"] != SELECTED:
        raise ValueError("Unexpected validation ranking")
    search = tables["search_run_metrics.csv"]
    if len(search) != 36:
        raise ValueError("Expected 36 search fits")
    for row in ranking:
        candidates = [r for r in search if r["architecture"] == row["architecture"]]
        if sorted(int(r["seed"]) for r in candidates) != [0, 1, 2]:
            raise ValueError("Expected search seeds 0, 1, 2 for every architecture")
        values = [float(r["best_validation_percent"]) for r in candidates]
        np.testing.assert_allclose([np.mean(values), np.std(values)],
                                  [float(row["validation_mean_percent"]),
                                   float(row["validation_population_sd_pp"])], rtol=0, atol=1e-10)
    expected_ranking = sorted(ranking, key=lambda r: (-float(r["validation_mean_percent"]),
                                                     int(r["binary_sites"]), r["architecture"]))
    if ranking != expected_ranking or [int(r["rank"]) for r in ranking] != list(range(1, 13)):
        raise ValueError("Ranking differs from validation-only selection and tie-breaking")
    resources = tables["resources.csv"]
    for field, total in (("signed_weights", 4428), ("binary_sites", 88560),
                         ("electronic_biases", 70), ("channel_products_per_image", 68256)):
        if sum(int(r[field]) for r in resources) != total:
            raise ValueError(f"Resource mismatch: {field}")
    for row in tables["depth_encoding.csv"]:
        pos, neg = .9 ** int(row["positive_written"]), .9 ** int(row["negative_written"])
        np.testing.assert_allclose([pos, neg, (pos-neg)/(1-.9**10)],
                                  [float(row[k]) for k in ("positive_transmission", "negative_transmission", "signed_weight")],
                                  rtol=0, atol=1e-12)
    return {"runs": results, "summary": summary, "mean_confusion_percent": mean_matrix.tolist(),
            "class_recall_range_percent": [float(np.diag(mean_matrix).min()), float(np.diag(mean_matrix).max())],
            "paired_accuracy_gain_pp": [metrics_gain(results, seed) for seed in SEEDS],
            "statistical_unit": "training seed; reused official test images; no ensemble",
            "standard_deviation": "population, ddof=0", "source_tables_validated": sorted(tables)}


def metrics_gain(results, seed):
    by_arch = {r["architecture"]: r["accuracy_percent"] for r in results if r["seed"] == seed}
    return by_arch[SELECTED] - by_arch[REFERENCE]


def plot_figures(tables, audit, output, red_text=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    blue, orange = "#187CC1", "#D76A26"

    def save(fig, name):
        if red_text:
            from matplotlib.text import Text
            for text in fig.findobj(Text):
                text.set_color("#FF0000")
        for ext in ("png", "svg", "pdf", "tiff"):
            options = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tiff" else {}
            fig.savefig(output / f"{name}.{ext}", dpi=600, facecolor="white", **options)
        plt.close(fig)

    levels = .9 ** np.arange(11)
    weights = (levels[:, None] - levels[None, :]) / (1-.9**10)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    axes[0].plot(np.arange(11), levels * 100, "o-", color=blue, ms=4)
    axes[0].set(xlabel="Written sites per branch", ylabel="Relative transmission (%)",
                xticks=range(0, 11, 2), ylim=(0, 106))
    axes[0].set_title("a  Assumed branch transmission", loc="left", weight="bold")
    im = axes[1].imshow(weights, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set(xlabel="Negative-branch written sites", ylabel="Positive-branch written sites",
                xticks=range(0, 11, 2), yticks=range(0, 11, 2))
    axes[1].set_title("b  Differential weight", loc="left", weight="bold")
    fig.colorbar(im, ax=axes[1], shrink=.85, label="Signed weight")
    save(fig, "Supplementary_Figure_12")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.5), layout="constrained")
    ax = axes[0, 0]
    ranking = tables["validation_ranking.csv"]
    for i, row in enumerate(ranking):
        color = orange if row["architecture"] == SELECTED else blue
        ax.errorbar(float(row["validation_mean_percent"]), i,
                    xerr=float(row["validation_population_sd_pp"]), fmt="o", color=color, ms=3, capsize=2)
    ax.set(yticks=range(12), yticklabels=[f"{r['channels']} / {r['kernel']} / {r['stride']}" for r in ranking],
           xlabel="Validation accuracy (%)", ylabel="C / K / stride", ylim=(11.6, -.6))
    ax.set_title("a  Validation ranking", loc="left", weight="bold")
    ax = axes[0, 1]
    for seed in SEEDS:
        values = [next(r["accuracy_percent"] for r in audit["runs"]
                       if r["architecture"] == arch and r["seed"] == seed) for arch in (REFERENCE, SELECTED)]
        ax.plot([0, 1], values, "o-", color="#B7BDC3", ms=4, lw=.8)
    for i, (arch, color) in enumerate(((REFERENCE, blue), (SELECTED, orange))):
        stats = audit["summary"][arch]["accuracy_percent"]
        ax.errorbar(i+.09, stats["mean"], yerr=stats["population_sd_pp"], fmt="s", color=color, ms=4, capsize=3)
    ax.set(xticks=[0, 1], xticklabels=["Reference", "Selected"], ylabel="Test accuracy (%)", xlim=(-.3, 1.3))
    ax.set_title("b  Paired confirmation", loc="left", weight="bold")
    ax = axes[1, 0]
    colors = [blue, orange, "#418B56"]
    for seed, color in zip(SEEDS, colors):
        history = [r for r in tables["selected_training_history.csv"] if int(r["seed"]) == seed and r["loss"]]
        ax.plot([int(r["epoch"]) for r in history], [float(r["loss"]) for r in history],
                color=color, label=f"Seed {seed}", lw=1.2)
    ax.set(xlabel="Epoch", ylabel="Mean cross-entropy", xticks=[1, 5, 10, 15, 20])
    ax.set_title("c  Sample-weighted training loss", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    matrix = np.array(audit["mean_confusion_percent"])
    im = ax.imshow(matrix, cmap="Blues", norm=PowerNorm(.3, vmin=0, vmax=100))
    for i in range(10):
        for j in range(10):
            if i == j or matrix[i, j] >= .5:
                ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=5.5,
                        color="white" if matrix[i, j] > 60 else "#142B3A",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .3} if red_text else None)
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit")
    ax.set_title("d  Mean row-normalised confusion", loc="left", weight="bold")
    fig.colorbar(im, ax=ax, shrink=.8, label="Percent", ticks=[0, 1, 10, 50, 100])
    save(fig, "Supplementary_Figure_13")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Supplementary_Data_1.zip or extracted directory")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--red-text", action="store_true", help="Mark regenerated figure text red for review")
    args = parser.parse_args()
    tables, members = load_tables(args.data, required_files=SOURCE_TABLES)
    audit = validate_tables(tables)
    audit["source_sha256"] = {n: hashlib.sha256(b).hexdigest() for n, b in members.items() if n.endswith(".csv")}
    plot_figures(tables, audit, args.output, args.red_text)
    (args.output / "numerical_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with (args.output / "Table_6.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables["confirmation_metrics.csv"][0]))
        writer.writeheader()
        writer.writerows(tables["confirmation_metrics.csv"])
    (args.output / "Table_7.csv").write_bytes(members["resources.csv"])
    print(json.dumps(audit["summary"], indent=2))


if __name__ == "__main__":
    main()
