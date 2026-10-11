"""Audit all 50 paired advisor tasks before producing descriptive summaries."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from numerical_simulation.advisor_experiments_20261004.run import (
    CASE_NAMES,
    build_case_dataset,
    load_formal_config,
)
from numerical_simulation.data_generation import generate_simulation_dataset, iter_test_batches
from numerical_simulation.metrics import compute_coefficient_metrics, compute_support_metrics
from tensor_abess.tensor.ops import support_from_tensor, tensor_inner


SEEDS = tuple(range(2026, 2036))
METHODS = ("adaptive", "fixed_known_s", "oracle")
CASE_LABELS = {
    "uniform_signed_snr2": "原始随机正负均匀幅度，SNR=2",
    "constant_signed_snr2": "等幅并保留原正负号，SNR=2",
    "constant_positive_snr2": "非零项全正等幅，SNR=2",
    "constant_positive_1_literal": "非零项字面值 +1，SNR≈219.486",
    "constant_positive_05_literal": "非零项字面值 +0.5，SNR≈54.871",
}
METHOD_LABELS = {"adaptive": "未知 s", "fixed_known_s": "已知 s 数量", "oracle": "Oracle 真 support"}
DISPLAY_METRICS = (
    "mode_f1_macro", "element_f1", "exact_recovery", "q_hat", "ree", "pmse", "r2_test",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"数据完整性核验失败：{message}")


def array_digest(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def close(actual: float, expected: float, message: str, tolerance: float = 1e-10) -> None:
    require(abs(float(actual) - float(expected)) <= tolerance, message)


def validate_support(support: np.ndarray, dimension: int, label: str) -> None:
    require(support.ndim == 1 and np.issubdtype(support.dtype, np.integer), f"{label} 必须为整数向量")
    require(np.array_equal(support, np.unique(support)), f"{label} 必须升序且无重复")
    require(bool(np.all((support >= 0) & (support < dimension))), f"{label} 超出合法索引范围")


def audit_tasks(output: Path) -> tuple[list[dict], dict]:
    expected_keys = {(seed, case) for seed in SEEDS for case in CASE_NAMES}
    tasks = {}
    for path in output.glob("seed_*.json"):
        task = json.loads(path.read_text(encoding="utf-8"))
        manifest = task["manifest"]
        key = (manifest["seed"], manifest["case"])
        require(key not in tasks, f"重复任务 {key}")
        require(path.name == f"seed_{key[0]}_{key[1]}.json", f"文件名和任务身份不一致：{path.name}")
        tasks[key] = task
    missing = sorted(expected_keys - set(tasks))
    extra = sorted(set(tasks) - expected_keys)
    require(not missing and not extra, f"需要完整 50 项；缺失={missing}；多余={extra}")
    config, manifest_path = load_formal_config()
    formal_config = json.loads(json.dumps(asdict(config)))
    require(config.n_train == 200 and config.n_test == 500, "训练/测试样本量不是 200/500")
    require(config.dims == (32, 20, 25) and config.sparsity == (4, 3, 5), "维度或真实稀疏度不符")
    require(config.rho == (0.3, 0.3, 0.3) and config.sigma == 1.0, "相关度或噪声标准差不符")
    require(config.k_max == 1 and config.max_cycles == 6 and config.snr == 2.0, "求解器参数或基准 SNR 不符")
    first_manifest = tasks[(SEEDS[0], CASE_NAMES[0])]["manifest"]
    source_hashes = first_manifest["source_fingerprints"]
    repository = Path(__file__).resolve().parents[2]
    for relative, digest in source_hashes.items():
        source_path = (repository / relative).resolve()
        require(source_path.is_relative_to(repository.resolve()), "源码指纹路径越出仓库")
        require(hashlib.sha256(source_path.read_bytes()).hexdigest() == digest, f"源码在运行后发生变化：{relative}")
    reference_environment = {key: first_manifest[key] for key in ("thread_environment", "python_version", "numpy_version")}
    require(set(reference_environment["thread_environment"].values()) == {"1"}, "实验线程配置不统一为 1")
    flat_rows = []
    metric_names = None
    all_keys = set()
    for seed in SEEDS:
        baseline = generate_simulation_dataset(config, seed=seed)
        noise = baseline.y_train - tensor_inner(baseline.X_train, baseline.B_true)
        test_X_hash = hashlib.sha256()
        test_noise_hash = hashlib.sha256()
        for X, y, signal in iter_test_batches(baseline, seed=seed + 1_000_000):
            test_X_hash.update(X.tobytes())
            test_noise_hash.update((y - signal).tobytes())
        expected_pairing = {
            "train_X_sha256": array_digest(baseline.X_train),
            "train_noise_sha256": array_digest(noise),
            "test_X_stream_sha256": test_X_hash.hexdigest(),
            "test_baseline_noise_stream_sha256": test_noise_hash.hexdigest(),
            **{f"support_{i}_sha256": array_digest(S) for i, S in enumerate(baseline.supports_true, 1)},
        }
        seed_noise = None
        seed_supports = None
        literal_snrs = {}
        for case in CASE_NAMES:
            task = tasks[(seed, case)]
            manifest = task["manifest"]
            stem = f"seed_{seed}_{case}"
            require(manifest["test_seed"] == seed + 1_000_000, f"{stem} 测试 seed 不符")
            require(manifest["base_config"] == formal_config, f"{stem} 基准参数发生变化")
            require(manifest["source_fingerprints"] == source_hashes, f"{stem} 源码指纹不统一")
            require(manifest["paired_input_fingerprints"] == expected_pairing, f"{stem} 训练/测试配对指纹不符")
            require({key: manifest[key] for key in reference_environment} == reference_environment, f"{stem} 计算环境不统一")
            expected_manifest_digest = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            require(task["manifest_sha256"] == expected_manifest_digest, f"{stem} manifest 校验码不符")
            equivalence = task["normalized_positive_1_vs_05_equivalence"]
            require(
                equivalence["identical_coefficients"] is True
                and equivalence["identical_training_response"] is True
                and equivalence["coefficient_max_abs_difference"] == 0.0,
                f"{stem} 归一化 +1/+0.5 等价核验未通过",
            )
            expected_dataset = build_case_dataset(baseline, case)
            require(manifest["case_config"] == json.loads(json.dumps(asdict(expected_dataset.config))), f"{stem} case 参数不符")
            close(task["actual_snr"], expected_dataset.signal_variance / config.sigma**2, f"{stem} 实际 SNR 不符")
            close(task["actual_snr"], manifest["case_config"]["snr"], f"{stem} 记录的 SNR 不一致")
            expected_snr = {
                "constant_positive_1_literal": 219.485911944,
                "constant_positive_05_literal": 54.871477986,
            }.get(case, 2.0)
            close(task["actual_snr"], expected_snr, f"{stem} 理论 SNR 不符", tolerance=1e-8)
            literal_snrs[case] = task["actual_snr"]
            require(manifest["case_y_sha256"] == array_digest(expected_dataset.y_train), f"{stem} 响应不能从原始配对数据重建")
            require(task["arrays_file"] == f"{stem}.npz", f"{stem} NPZ 文件名不符")
            with np.load(output / task["arrays_file"], allow_pickle=False) as saved:
                B_true = saved["B_true"]
                require(B_true.shape == config.dims and bool(np.isfinite(B_true).all()), f"{stem} 真实系数形状/有限性错误")
                require(np.array_equal(B_true, expected_dataset.B_true), f"{stem} 真实系数不能逐元素重建")
                require(manifest["case_B_sha256"] == array_digest(B_true), f"{stem} 真实系数指纹不符")
                saved_noise = saved["train_noise"]
                require(saved_noise.shape == (config.n_train,) and np.array_equal(saved_noise, noise), f"{stem} 原始训练噪声不符")
                true_supports = tuple(saved[f"S{i}_true"] for i in (1, 2, 3))
                for i, (S, dimension, truth) in enumerate(zip(true_supports, config.dims, baseline.supports_true), 1):
                    validate_support(S, dimension, f"{stem} S{i}_true")
                    require(np.array_equal(S, truth), f"{stem} true support 与原数据不符")
                require(all(np.array_equal(S, actual) for S, actual in zip(true_supports, support_from_tensor(B_true))), f"{stem} B_true/support 不符")
                require(np.count_nonzero(B_true) == 60, f"{stem} 非零系数数量不是 60")
                if seed_noise is not None:
                    require(np.array_equal(seed_noise, saved_noise), f"seed {seed} 各 case 噪声不完全配对")
                    require(all(np.array_equal(first, S) for first, S in zip(seed_supports, true_supports)), f"seed {seed} 各 case support 不完全配对")
                seed_noise = saved_noise.copy()
                seed_supports = tuple(S.copy() for S in true_supports)
                require(len(task["results"]) == 3 and {row["method"] for row in task["results"]} == set(METHODS), f"{stem} 方法不完整或重复")
                for row in task["results"]:
                    method = row["method"]
                    key = (seed, case, method)
                    require(key not in all_keys, f"重复方法结果 {key}")
                    all_keys.add(key)
                    metrics = row["metrics"]
                    require(bool(np.isfinite(list(metrics.values())).all()), f"{key} 指标非有限值")
                    if metric_names is None:
                        metric_names = tuple(metrics)
                    require(set(metrics) == set(metric_names), f"{key} 指标集合不统一")
                    B_hat = saved[f"B_{method}"]
                    estimated_supports = tuple(saved[f"S{i}_{method}"] for i in (1, 2, 3))
                    require(B_hat.shape == config.dims and bool(np.isfinite(B_hat).all()), f"{key} 拟合系数形状/有限性错误")
                    for i, (S, dimension) in enumerate(zip(estimated_supports, config.dims), 1):
                        validate_support(S, dimension, f"{key} S{i}")
                    selected_mask = np.zeros(config.dims, dtype=bool)
                    selected_mask[np.ix_(*estimated_supports)] = True
                    require(np.count_nonzero(B_hat[~selected_mask]) == 0, f"{key} 系数在拟合 support 外非零")
                    recomputed = {
                        **compute_support_metrics(*estimated_supports, *true_supports),
                        **compute_coefficient_metrics(B_hat, B_true),
                    }
                    for metric, value in recomputed.items():
                        close(metrics[metric], value, f"{key} {metric} 与 NPZ 重算不符")
                    if method == "fixed_known_s":
                        require(tuple(len(S) for S in estimated_supports) == config.sparsity, f"{key} 已知 s 尺寸不符")
                        require(metrics["q_hat"] == 60.0, f"{key} 已知 s 的 q 不是 60")
                    if method == "oracle":
                        require(metrics["exact_recovery"] == 1.0, f"{key} Oracle 未使用真 support")
                    flat_rows.append({
                        "seed": seed, "case": case, "method": method,
                        "actual_snr": task["actual_snr"], "manifest_sha256": task["manifest_sha256"],
                        **metrics,
                    })
        close(literal_snrs["constant_positive_1_literal"] / literal_snrs["constant_positive_05_literal"], 4.0, f"seed {seed} literal SNR 比不是 4")
    require(len(flat_rows) == 150 and len(all_keys) == 150, "总结果数不是 150")
    counts = {(case, method): sum(row["case"] == case and row["method"] == method for row in flat_rows) for case in CASE_NAMES for method in METHODS}
    require(set(counts.values()) == {10}, "各 case×method 并非恰好 10 个 seed")
    return flat_rows, {
        "task_count": len(tasks), "method_result_count": len(flat_rows),
        "seeds": SEEDS, "cases": CASE_NAMES, "methods": METHODS,
        "all_integrity_checks_passed": True,
        "source_fingerprints": source_hashes,
        "base_manifest_file": str(manifest_path.relative_to(repository)),
        "metric_names": metric_names,
    }


def historical_audit(rows: list[dict], repository: Path) -> dict:
    row_index = {(row["seed"], row["case"], row["method"]): row for row in rows}
    records = []
    for version, method, strict in (("v5_adaptive", "adaptive", True), ("v3", "fixed_known_s", False)):
        path = repository / f"numerical_simulation/outputs_full_{version}/raw_results.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            history = [row for row in csv.DictReader(handle) if row["experiment"] == "baseline" and row["method"] == "Tensor-ABESS"]
        require(len(history) == 10 and {int(row["seed"]) for row in history} == set(SEEDS), f"历史 {version} baseline seed 不完整")
        for old in history:
            seed = int(old["seed"])
            current = row_index[(seed, "uniform_signed_snr2", method)]
            differences = {
                metric: abs(float(current[metric]) - float(old[metric]))
                for metric in current.keys() - {"seed", "case", "method", "actual_snr", "manifest_sha256", "runtime_sec"}
            }
            matches = {metric: difference <= 1e-10 for metric, difference in differences.items()}
            if strict:
                require(all(matches.values()), f"seed {seed} v5 复现偏差超过 1e-10：{differences}")
            records.append({
                "historical_version": version, "method": method, "seed": seed,
                "required_to_pass": strict, "all_metrics_match": all(matches.values()),
                "metric_absolute_differences": differences, "metric_matches_at_1e_10": matches,
            })
    return {
        "absolute_tolerance": 1e-10,
        "excluded_metric": "runtime_sec (elapsed time changes with process scheduling)",
        "historical_adaptive_v5_all_passed": all(row["all_metrics_match"] for row in records if row["required_to_pass"]),
        "historical_fixed_v3_all_match": all(row["all_metrics_match"] for row in records if not row["required_to_pass"]),
        "historical_fixed_v3_is_descriptive_only": True,
        "records": records,
    }


def summarize_rows(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    summaries = []
    for case in CASE_NAMES:
        for method in METHODS:
            members = [row for row in rows if row["case"] == case and row["method"] == method]
            summary = {"case": case, "method": method, "n_reps": len(members), "actual_snr": members[0]["actual_snr"]}
            for metric in metrics:
                values = np.array([row[metric] for row in members], dtype=float)
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sample_sd"] = float(values.std(ddof=1))
            summary["exact_recovery_successes"] = int(sum(row["exact_recovery"] for row in members))
            summary["exact_recovery_denominator"] = 10
            summaries.append(summary)
    return summaries


def historical_scenario_comparison(repository: Path) -> tuple[list[dict], dict]:
    """Read existing v3/v5 evidence; this does not extend the new baseline run."""
    source_paths = {
        "fixed": repository / "numerical_simulation/outputs_full_v3/raw_results.csv",
        "adaptive": repository / "numerical_simulation/outputs_full_v5_adaptive/raw_results.csv",
    }
    indexes = {}
    key_fields = ("experiment", "parameter", "scenario_value", "scenario_order", "replicate", "seed")
    for method, path in source_paths.items():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = [row for row in csv.DictReader(handle) if row["method"] == "Tensor-ABESS"]
        index = {tuple(row[field] for field in key_fields): row for row in records}
        require(len(records) == 140 and len(index) == 140, f"历史 {method} 并非唯一的 140 项")
        indexes[method] = index
    require(set(indexes["fixed"]) == set(indexes["adaptive"]), "历史 14 场景配对身份不一致")
    scenario_keys = sorted({key[:4] for key in indexes["fixed"]})
    require(len(scenario_keys) == 14, "历史场景数量不是 14")
    summary_rows = []
    metadata_fields = (
        "n_train", "n_test", "p1", "p2", "p3", "s1_true", "s2_true", "s3_true",
        "rho_1", "rho_2", "rho_3", "snr", "sigma", "support_shape",
    )
    for scenario in scenario_keys:
        keys = sorted(key for key in indexes["fixed"] if key[:4] == scenario)
        require(len(keys) == 10, f"历史场景 {scenario} 不是 10 次")
        pairs = [(indexes["fixed"][key], indexes["adaptive"][key]) for key in keys]
        for fixed, adaptive in pairs:
            require(all(fixed[field] == adaptive[field] for field in metadata_fields), f"历史场景 {scenario} 数据参数不配对")
        result = {
            "experiment": scenario[0], "parameter": scenario[1],
            "scenario_value": scenario[2], "scenario_order": int(scenario[3]), "n_pairs": 10,
            "fixed_source": str(source_paths["fixed"].relative_to(repository)),
            "adaptive_source": str(source_paths["adaptive"].relative_to(repository)),
            "source_code_hash_available": False,
            "interpretation": "historical full-pipeline comparison; differing initialization and incomplete historical code hashes",
            **{field: pairs[0][0][field] for field in metadata_fields},
        }
        for metric in DISPLAY_METRICS:
            fixed_values = np.array([float(first[metric]) for first, _ in pairs])
            adaptive_values = np.array([float(second[metric]) for _, second in pairs])
            require(bool(np.isfinite(fixed_values).all() and np.isfinite(adaptive_values).all()), f"历史 {scenario} {metric} 非有限")
            difference = fixed_values - adaptive_values
            result[f"fixed_{metric}_mean"] = float(fixed_values.mean())
            result[f"fixed_{metric}_sample_sd"] = float(fixed_values.std(ddof=1))
            result[f"adaptive_{metric}_mean"] = float(adaptive_values.mean())
            result[f"adaptive_{metric}_sample_sd"] = float(adaptive_values.std(ddof=1))
            result[f"fixed_minus_adaptive_{metric}_mean"] = float(difference.mean())
            result[f"fixed_minus_adaptive_{metric}_sample_sd"] = float(difference.std(ddof=1))
            if metric == "exact_recovery":
                result["fixed_exact_successes"] = int(fixed_values.sum())
                result["adaptive_exact_successes"] = int(adaptive_values.sum())
            if metric == "pmse":
                result["fixed_pmse_better_count"] = int((difference < -1e-10).sum())
                result["fixed_pmse_worse_count"] = int((difference > 1e-10).sum())
                result["pmse_equal_count_at_1e_10"] = int((np.abs(difference) <= 1e-10).sum())
        summary_rows.append(result)
    aggregate = {
        "scenario_count": 14, "pair_count": 140,
        "fixed_pmse_better_count": sum(row["fixed_pmse_better_count"] for row in summary_rows),
        "fixed_pmse_worse_count": sum(row["fixed_pmse_worse_count"] for row in summary_rows),
        "pmse_equal_count_at_1e_10": sum(row["pmse_equal_count_at_1e_10"] for row in summary_rows),
        "fixed_exact_successes": sum(row["fixed_exact_successes"] for row in summary_rows),
        "adaptive_exact_successes": sum(row["adaptive_exact_successes"] for row in summary_rows),
        "historical_code_hashes_incomplete": True,
        "descriptive_only_not_a_new_full_suite": True,
    }
    return summary_rows, aggregate


def paired_differences(rows: list[dict], metrics: tuple[str, ...]) -> list[dict]:
    index = {(row["seed"], row["case"], row["method"]): row for row in rows}
    contrasts = [("fixed_minus_adaptive", case, "fixed_known_s", case, "adaptive") for case in CASE_NAMES]
    coefficient_pairs = (
        ("signed_constant_minus_uniform", "constant_signed_snr2", "uniform_signed_snr2"),
        ("positive_constant_minus_uniform", "constant_positive_snr2", "uniform_signed_snr2"),
        ("positive_constant_minus_signed_constant", "constant_positive_snr2", "constant_signed_snr2"),
    )
    contrasts += [(name, first, method, second, method) for name, first, second in coefficient_pairs for method in METHODS]
    results = []
    for name, first_case, first_method, second_case, second_method in contrasts:
        for metric in metrics:
            differences = np.array([
                index[(seed, first_case, first_method)][metric] - index[(seed, second_case, second_method)][metric]
                for seed in SEEDS
            ], dtype=float)
            results.append({
                "contrast": name, "first_case": first_case, "first_method": first_method,
                "second_case": second_case, "second_method": second_method, "metric": metric,
                "n_pairs": 10, "difference_mean": float(differences.mean()),
                "difference_sample_sd": float(differences.std(ddof=1)),
                "difference_min": float(differences.min()), "difference_max": float(differences.max()),
                **{f"difference_seed_{seed}": float(value) for seed, value in zip(SEEDS, differences)},
            })
    return results


def report_text(summaries: list[dict], audit: dict, history: dict) -> str:
    index = {(row["case"], row["method"]): row for row in summaries}

    def statistic(case: str, method: str, metric: str) -> str:
        row = index[(case, method)]
        if metric == "exact_recovery":
            return f"{row['exact_recovery_successes']}/10"
        return f"{row[f'{metric}_mean']:.4f} ± {row[f'{metric}_sample_sd']:.4f}"

    lines = [
        "导师建议补充实验：完整结果与核验记录（2026-10-04）", "",
        "一、是否有必要执行",
        "两项建议都值得做：等幅系数可检查原始幅度异质性是否增加难度；已知 s 可检查真实稀疏度数量已给定时的表现。采用配对实验并保留原算法，不预设一定改善。", "",
        "二、数据与实验设置",
        "训练 n=200，独立测试 n=500；张量维度 32×20×25（16000 个潜在系数）；真 mode support 大小 s=(4,3,5)，矩形真 support 内有 60 个非零元素，support 外始终为 0。",
        "各 mode AR(1) 相关度 rho=0.3，噪声标准差 sigma=1；连续 support；10 个 seed=2026–2035，测试 seed=训练 seed+1000000，测试 batch size=100。",
        "精确复用 v5 checkpoint：k_max=1、max_cycles=6，BIC 初始化 c=3、20 个 lambda、max_iter=500、tol=1e-6；adaptive 使用既有 MSIC 参数 alpha=1.5、gamma=1、one_se_scale=1。",
        "各 seed 通过原始 generate_simulation_dataset 生成同一份可复现数据，五组共用 X、真 support、训练噪声；重建 B 后只重算 y=<X,B>+原噪声，测试流也共用 X 和噪声。全部运算环境与单线程设置一致。",
        "原始设定不是每个非零元素都在 1 附近：先生成随机正负号乘 U(0.8,1.2)，然后整体缩放到 SNR=2。", "",
        "三、五个系数组与算法",
    ]
    lines.extend(f"{number}. {case}：{CASE_LABELS[case]}。" for number, case in enumerate(CASE_NAMES, 1))
    lines.extend([
        "前三组都保持 SNR=2：signed 等幅只去掉幅度异质性，全正等幅同时改变符号结构。固定同 SNR 时，先设 +1 与先设 +0.5 再整体归一化得到完全相同的 B 和 y，所以只拟合一次并保存逐元素等价核验。",
        "全正等幅的真 support 系数块同时变成秩 1，因此它与随机符号组的差异也涉及系数方向及低秩结构，不能只解释为去掉幅度波动。",
        "字面值 +1/+0.5 两组不整体缩放，sigma=1 不变；其 SNR 分别为 219.485911944 和 54.871477986，比例为 4。这两组表现变化同时包含信号增强，不能全部解释成等幅系数的好处。",
        "每组比较未知 s 的 adaptive、只知道每个 mode 选择数量 (4,3,5) 的 fixed，以及知道真 support 位置的 Oracle OLS；fixed 仍需自己寻找索引，并非 Oracle。", "",
        "四、各组完整指标表",
        "除 Exact 外，各值为 10 次均值 ± 样本标准差（ddof=1）；Exact 为精确恢复次数/10。F1、element F1、Exact、R2 越高越好，REE、PMSE 越低越好；q 为估计系数数量。",
        "分组 | 方法 | Mode F1 | Element F1 | Exact | q | REE | PMSE | R2",
    ])
    for case in CASE_NAMES:
        for method in METHODS:
            lines.append(" | ".join([CASE_LABELS[case], METHOD_LABELS[method], *(statistic(case, method, metric) for metric in DISPLAY_METRICS)]))
    lines.extend(["", "五、根据本次数字形成的判断"])
    base = index[("uniform_signed_snr2", "adaptive")]
    signed = index[("constant_signed_snr2", "adaptive")]
    positive = index[("constant_positive_snr2", "adaptive")]
    for label, comparator in (("去掉幅度异质性但保留随机正负号", signed), ("非零项改为全正等幅", positive)):
        f1_delta = comparator["mode_f1_macro_mean"] - base["mode_f1_macro_mean"]
        pmse_delta = comparator["pmse_mean"] - base["pmse_mean"]
        f1_direction = "提高" if f1_delta > 0 else "降低" if f1_delta < 0 else "持平"
        pmse_direction = "增加" if pmse_delta > 0 else "减少" if pmse_delta < 0 else "持平"
        lines.append(
            f"同 SNR=2 下，{label}：adaptive Mode F1 {f1_direction} {abs(f1_delta):.4f}，"
            f"PMSE {pmse_direction} {abs(pmse_delta):.4f}；Exact 为 {comparator['exact_recovery_successes']}/10，原设定为 {base['exact_recovery_successes']}/10。"
        )
    lines.append("已知 s 相对未知 s 的配对均值差（fixed−adaptive）：")
    for case in CASE_NAMES:
        fixed = index[(case, "fixed_known_s")]
        adaptive = index[(case, "adaptive")]
        f1_delta = fixed["mode_f1_macro_mean"] - adaptive["mode_f1_macro_mean"]
        pmse_delta = fixed["pmse_mean"] - adaptive["pmse_mean"]
        lines.append(
            f"{CASE_LABELS[case]}：ΔMode F1={f1_delta:+.4f}，ΔPMSE={pmse_delta:+.4f}；"
            f"fixed Exact={fixed['exact_recovery_successes']}/10，adaptive Exact={adaptive['exact_recovery_successes']}/10。"
        )
    f1_better = sum(index[(case, "fixed_known_s")]["mode_f1_macro_mean"] > index[(case, "adaptive")]["mode_f1_macro_mean"] for case in CASE_NAMES)
    pmse_better = sum(index[(case, "fixed_known_s")]["pmse_mean"] < index[(case, "adaptive")]["pmse_mean"] for case in CASE_NAMES)
    lines.append(f"五组中，fixed 的平均 Mode F1 在 {f1_better} 组更高、平均 PMSE 在 {pmse_better} 组更低；不同指标应分别判断，不能预先假定已知 s 必然全面更好。")
    lines.extend([
        "以上为 10 次配对重复的描述性结果，未作显著性检验或显著性宣称；paired_differences.csv 同时保存每个 seed 的差值及均值/样本标准差。", "",
        "六、完整性与历史复现",
        f"{audit['task_count']} 个 seed×case 完整任务，{audit['method_result_count']} 个方法结果，15 个 case×method 组合均为 10 次；所有指标有限。",
        "全部 task 的代码指纹一致且与当前原源码一致，manifest 校验码通过；逐 seed 重建原 X、support、噪声及独立测试流指纹一致；NPZ 系数逐元素重建、support/REE 重算与 JSON 一致。",
        "fixed 的三个估计 support 大小始终是 (4,3,5)、q=60；Oracle Exact 始终为 1；全部归一化 +1/+0.5 等价核验通过，三组 normalized SNR=2、两组 literal SNR 理论值与比例通过。",
        "原设定 adaptive 的 10 个 seed 已与 outputs_full_v5_adaptive 历史 baseline 比较，除运行耗时外保存的全部共同指标绝对偏差不超过 1e-10。",
        f"原设定 fixed 与 outputs_full_v3 历史 baseline 的共同指标{'全部一致（绝对偏差≤1e-10）' if history['historical_fixed_v3_all_match'] else '存在差异'}；该比较仅作追溯说明，不将不完整的历史代码版本作为本次有效性前置。逐指标结果见 reproduction_audit.json。",
        f"历史 14 场景的 140 次配对中，fixed 的 PMSE 更低/更高/相同次数分别为 {history['all_scenarios']['fixed_pmse_better_count']}/{history['all_scenarios']['fixed_pmse_worse_count']}/{history['all_scenarios']['pmse_equal_count_at_1e_10']}；Exact 成功数 fixed={history['all_scenarios']['fixed_exact_successes']}/140，adaptive={history['all_scenarios']['adaptive_exact_successes']}/140。详见 historical_fixed_vs_adaptive.csv；此项只读既有历史结果，并非本次重新训练 14 场景。", "",
        "七、解释边界与后续补充",
        "现有 fixed 路径使用 marginal top support 初始化，adaptive 使用 GLasso 初始化并进行 MSIC 搜索；因此本次比较是两条既有完整算法流程，无法把差异单独归因于是否知道 s。未修改核心算法以强制统一初始化。",
        "本次只补正式 baseline 参数下的五组系数×两条 Tensor-ABESS 路径和 Oracle，未重跑原报告的所有 14 个场景，也未声称所有 SNR、维度、样本量与相关度条件都成立。",
        "后续先根据同 SNR 三组的效应方向决定是否扩展更多重复与关键难度场景；如需单独识别稀疏度选择贡献，可另行设计初始化受控实验。保留 literal 两组作为强信号补充，不替代 SNR=2 的公平对照。", "",
        "八、来源与文件",
        "Presentation/数值模拟汇报_v5_beamer.tex；numerical_simulation/outputs_full_v5_adaptive/checkpoint_manifest.json 与 raw_results.csv；numerical_simulation/outputs_full_v3/raw_results.csv。",
        "本目录 run.py/test_run.py：独立运行与数据配对测试；summarize.py：完整性及历史复现审核；outputs/seed_*.json 与 seed_*.npz：完整任务证据。",
        "raw_results.csv：150 行方法结果；summary_results.csv：15 行均值/样本标准差；paired_differences.csv：固定−自适应及三项同 SNR 系数对照；reproduction_audit.json：完整性与历史指标追溯；historical_fixed_vs_adaptive.csv：既有 14 场景对照；RESULTS.txt：本报告。",
    ])
    return "\n".join(lines) + "\n"


def csv_text(rows: list[dict]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def save_exclusive(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name, suffix=".tmp", encoding="utf-8", newline="", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    os.rename(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args(argv)
    filenames = ("raw_results.csv", "summary_results.csv", "paired_differences.csv", "reproduction_audit.json", "historical_fixed_vs_adaptive.csv", "RESULTS.txt")
    require(args.output.is_dir(), "任务输出目录不存在")
    for filename in filenames:
        if (args.output / filename).exists():
            raise FileExistsError(f"拒绝覆写已有汇总：{args.output / filename}")
    rows, audit = audit_tasks(args.output)
    repository = Path(__file__).resolve().parents[2]
    history = historical_audit(rows, repository)
    historical_scenarios, aggregate = historical_scenario_comparison(repository)
    history["all_scenarios"] = aggregate
    summaries = summarize_rows(rows, audit["metric_names"])
    differences = paired_differences(rows, audit["metric_names"])
    payloads = {
        "raw_results.csv": csv_text(rows),
        "summary_results.csv": csv_text(summaries),
        "paired_differences.csv": csv_text(differences),
        "historical_fixed_vs_adaptive.csv": csv_text(historical_scenarios),
        "reproduction_audit.json": json.dumps({"integrity": audit, "history": history}, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        "RESULTS.txt": report_text(summaries, audit, history),
    }
    for filename, content in payloads.items():
        save_exclusive(args.output / filename, content)
    print(f"完整性核验通过：50 个任务，150 个结果；v5 baseline 的 10 个 seed 均复现。汇总已写入 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
