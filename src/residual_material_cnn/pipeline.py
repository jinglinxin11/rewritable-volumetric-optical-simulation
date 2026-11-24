"""Reproducible train/compile/evaluate workflow; never promotes a formal baseline."""

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .artifacts import export_masks, save_compiled, save_json
from .config import TrainingConfig, configuration
from .metrics import error_metrics, legacy_mac_benchmark
from .network import compile_network
from .training import evaluate, train_reference

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def layer_errors(reference, compiled, patches):
    """Separate fixed-span linear MAC error from bias, activation and classifier metrics."""
    results = {}
    inputs = patches.double()
    for name, layer in reference.layers.items():
        weight = layer.weight.detach().double()
        scale = inputs.amax(-1, keepdim=True).clamp_min(1.0)
        illumination = inputs / scale
        target = F.linear(illumination, weight)
        actual = compiled.layers[name](illumination)
        # As in the historical fixture, compare mean products in a fixed [-1,1] interval.
        fan_in = weight.shape[1]
        results[name] = dict(
            input_terms=fan_in,
            normalization="linear dot product / fan_in",
            bias_included=False,
            sample_vectors=int(illumination.numel() / fan_in),
            **error_metrics((target / fan_in).numpy(), (actual / fan_in).numpy()),
        )
        value = F.linear(
            inputs, weight, None if layer.bias is None else layer.bias.detach().double()
        )
        if name == "conv":
            inputs = F.adaptive_avg_pool2d(
                F.relu(value.transpose(1, 2).reshape(-1, 3, 10, 10)), 3
            ).flatten(1)
        elif name != "fc3":
            inputs = F.relu(value)
    return results


def source_hashes():
    package = Path(__file__).resolve().parent
    paths = list(package.glob("*.py")) + list((package / "assets").rglob("*.npz"))
    return {
        p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def run_training(data, output, seeds=(0, 1, 2), epochs=20, batch_size=512, learning_rate=0.005):
    output = Path(output)
    seeds = tuple(seeds)
    TrainingConfig(seeds, epochs, batch_size, learning_rate)
    before = source_hashes()
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "manifest.json",
        dict(
            **configuration(),
            seeds=seeds,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning="offline float weights clipped to [-1,1], followed by residual3 compilation",
            checkpoint_selection="earliest highest validation; all seeds locked before test",
            source_sha256=before,
            data=data.provenance,
            torch_version=str(torch.__version__),
            numpy_version=np.__version__,
            device="cpu",
        ),
    )
    save_json(output / "legacy_clean_mac_benchmark.json", legacy_mac_benchmark())
    selected, models = [], []
    for seed in seeds:
        folder = output / f"seed{seed}"
        folder.mkdir()
        reference, record = train_reference(data, seed, folder, epochs, batch_size, learning_rate)
        compiled = compile_network(reference)
        save_compiled(folder / "compiled.npz", compiled)
        export_masks(compiled, folder / "write_masks")
        models.append((reference, compiled))
        selected.append(record)
    save_json(output / "locked_before_test.json", selected)
    records = []
    for selected_row, (reference, compiled) in zip(selected, models, strict=True):
        seed = selected_row["seed"]
        ref_eval = evaluate(reference, *data.test)
        comp_eval = evaluate(compiled, *data.test)
        disagreements = int((ref_eval["predictions"] != comp_eval["predictions"]).sum())
        target = np.concatenate(
            [x.weight.detach().double().numpy().ravel() for x in reference.layers.values()]
        )
        actual = np.concatenate(
            [x.encoded_matrix().effective_weights().ravel() for x in compiled.layers.values()]
        )
        record = dict(
            **selected_row,
            reference_test_accuracy_pct=ref_eval["accuracy_pct"],
            compiled_test_accuracy_pct=comp_eval["accuracy_pct"],
            differing_test_predictions=disagreements,
            coefficient_error=error_metrics(target, actual),
            clean_layer_mac_errors=layer_errors(reference, compiled, data.validation[0][:256]),
        )
        records.append(record)
        folder = output / f"seed{seed}"
        save_json(folder / "metrics.json", record)
        np.savez_compressed(
            folder / "test_predictions.npz",
            labels=data.test[1].numpy(),
            reference=ref_eval["predictions"].numpy(),
            compiled=comp_eval["predictions"].numpy(),
        )
        LOGGER.info(
            "seed%s: reference=%.2f%%, residual3=%.2f%%",
            seed,
            ref_eval["accuracy_pct"],
            comp_eval["accuracy_pct"],
        )
    ref = [r["reference_test_accuracy_pct"] for r in records]
    comp = [r["compiled_test_accuracy_pct"] for r in records]
    summary = dict(
        reference_test_mean_pct=float(np.mean(ref)),
        reference_population_sd_pp=float(np.std(ref)),
        compiled_test_mean_pct=float(np.mean(comp)),
        compiled_population_sd_pp=float(np.std(comp)),
        seeds=seeds,
        records=records,
        resources=configuration()["resources"],
        limitations=[
            "ideal simulation, not hardware",
            "reused MNIST test cohort",
            "new no-convolution-bias residual3 model; historical accuracies do not transfer",
        ],
    )
    save_json(output / "summary.json", summary)
    if before != source_hashes():
        raise RuntimeError("A source file or retained formal artifact changed during the run")
    save_json(
        output / "verification.json", dict(frozen_inputs_unchanged=True, seeds_completed=len(seeds))
    )
    lines = [
        "# 三核无卷积偏置 + 三组残差：本轮结果",
        "",
        "全连接电子偏置保留。浮点离线训练后编译，理想无噪声强度模型；不是旧有偏置模型的原样复现。",
        "",
        "| seed | 浮点参考测试 % | 三组残差测试 % | 预测不同数 |",
        "|---|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['seed']} | {r['reference_test_accuracy_pct']:.2f} | "
            f"{r['compiled_test_accuracy_pct']:.2f} | {r['differing_test_predictions']} |"
        )
    lines.extend(
        [
            "",
            f"三组残差准确率：**{np.mean(comp):.2f}% ± {np.std(comp):.2f} pp**（种子间总体标准差）。",
            "",
            "18.55 bit 对应另外的 16 输入、8 输出随机乘加基准；不能当作该网络或硬件位数。",
            "逐层 bit 明确使用固定跨度 2、点积除以 fan-in 的归一化，仅计线性乘加误差。",
        ]
    )
    (output / "REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
