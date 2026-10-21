"""Task 9.1/9.2: read-only reanalysis of the sealed 250 paired results.

Run from repository root: .venv/Scripts/python.exe -B research/taskcard9/attribution.py
No fits, data generation, predictions, merges, or writes to historical assets.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def save_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                         for k, v in row.items()} for row in rows)


def distribution(values):
    values = np.asarray(values, dtype=float)
    assert values.size and np.isfinite(values).all()
    positives = sorted((float(v) for v in values if v > 1e-12), reverse=True)
    net, gross = float(values.sum()), sum(positives)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p05=float(np.quantile(values, .05)), p90=float(np.quantile(values, .9)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()),
                minimum=float(values.min()), improved=int(sum(values < -1e-12)),
                tied=int(sum(abs(values) <= 1e-12)), worsened=int(sum(values > 1e-12)),
                net_sum=net, gross_positive_sum=gross,
                top5_positive_share_of_gross=sum(positives[:5])/gross if gross else None,
                top10_positive_share_of_gross=sum(positives[:10])/gross if gross else None,
                top5_positive_share_of_net=sum(positives[:5])/net if abs(net) > 1e-12 else None,
                top10_positive_share_of_net=sum(positives[:10])/net if abs(net) > 1e-12 else None)


def support_relation(left, right):
    same_shape = list(map(len, left)) == list(map(len, right))
    return {True: "same_shape_different_support", False: "different_shape"}[same_shape] if left != right else "same_support"


def direction(value):
    return "increase" if value > 1e-12 else ("decrease" if value < -1e-12 else "tie")


def load_audit(path, name, root):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.audit(root) if name == "cluster_audit" else module.audit()


def collect(root):
    cluster = root / "numerical_simulation/outputs_v51_attribution/cluster_run"
    plan = read(cluster / "plan.json")
    names = {0: "Baseline", 1: "SNR0.5", 5: "n100", 7: "n400", 11: "rho0"}
    folders = dict(v5=root / "numerical_simulation/outputs_known_s_full_50",
                   v51=cluster, v52=root / "numerical_simulation/outputs_v52_search_final_full")
    models, pairs, assets = [], [], {}
    for task in plan["tasks"]:
        name = f"task_{task['task_id']:06d}"
        with np.load(folders["v5"] / "tasks" / (name + ".npz"), allow_pickle=False) as saved:
            truth = saved["B_true"].copy()
            true_support = [saved[f"S{i}_true"].tolist() for i in (1, 2, 3)]
        selected, coefficients = {}, {}
        for version, folder in folders.items():
            path = folder / "tasks" / (name + ".json")
            record = read(path)
            result = record if version == "v51" else next(r for r in record["results"] if r["method"] == "adaptive")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
                suffix = "" if version == "v51" else "_adaptive"
                B = saved["B" + suffix].copy()
                supports = [saved[f"S{i}{suffix}"].tolist() for i in (1, 2, 3)]
                assert not any(k in saved.files for k in ("X", "y", "X_train", "y_train", "X_test", "y_test", "candidate_coefficients"))
            for artifact in (path, path.with_suffix(".npz")):
                assets[artifact.relative_to(root).as_posix()] = dict(bytes=artifact.stat().st_size, sha256=sha(artifact))
            caps = result["trace"]["s_max"]
            grids = read(cluster / "tasks" / (name + ".json"))["grids"]
            metric = result["metrics"]
            shape = list(map(len, supports))
            q = math.prod(shape)
            capacity = all(s <= c for s, c in zip(map(len, true_support), caps))
            grid_key = "truth_shape_in_v52" if version == "v52" else "truth_shape_in_v51"
            row = dict(task_id=task["task_id"], scenario=names[task["scenario_id"]],
                       scenario_id=task["scenario_id"], replicate=task["replicate"], seed=task["seed"],
                       test_seed=task["test_seed"], input_key=task["input_key"], version=version,
                       n=task["config"]["n_train"], snr=task["config"]["snr"], rho=task["config"]["rho"],
                       shape=shape, support=supports, q=q, q_over_Q=q/math.floor(.7*task["config"]["n_train"]),
                       Q=math.floor(.7*task["config"]["n_train"]), caps=caps,
                       true_shape=list(map(len, true_support)), true_support=true_support,
                       true_q=math.prod(map(len, true_support)), true_q_within_Q=math.prod(map(len, true_support)) <= math.floor(.7*task["config"]["n_train"]),
                       truth_shape_within_caps=capacity, truth_shape_in_fixed_grid=grids[grid_key],
                       fixed_grid_count=grids["v52_count" if version == "v52" else "v51_count"],
                       actual_adaptive_candidate_truth_hit="not available", all_pool_truth_hit="not available",
                       ols_rank="not available", ols_condition="not available",
                       pmse=metric["pmse"], rss=metric["rss_train"], msic=result["sic"],
                       ree=metric["ree"], mode_f1=metric["mode_f1_macro"], element_f1=metric["element_f1"],
                       exact=int(metric["exact_recovery"]),
                       B_sha256=hashlib.sha256(np.ascontiguousarray(B).tobytes()).hexdigest())
            assert math.isclose(np.linalg.norm(B-truth)/np.linalg.norm(truth), row["ree"], abs_tol=1e-12)
            assert row["exact"] == int(supports == true_support)
            models.append(row)
            selected[version], coefficients[version] = row, B
        for phase, left, right in (("BCD", "v5", "v51"), ("Grid", "v51", "v52")):
            a, b = selected[left], selected[right]
            delta = b["pmse"]-a["pmse"]
            qdelta = b["q"]-a["q"]
            relation = support_relation(a["support"], b["support"])
            pairs.append(dict(task_id=task["task_id"], scenario=a["scenario"], scenario_id=a["scenario_id"],
                              replicate=task["replicate"], seed=task["seed"], test_seed=task["test_seed"], phase=phase,
                              left_version=left, right_version=right, left_pmse=a["pmse"], right_pmse=b["pmse"],
                              delta_pmse=delta, outcome=direction(delta), left_q=a["q"], right_q=b["q"],
                              delta_q=qdelta, q_direction=direction(qdelta), left_shape=a["shape"], right_shape=b["shape"],
                              shape_delta=[r-l for l,r in zip(a["shape"], b["shape"])], left_support=a["support"],
                              right_support=b["support"], support_relation=relation, support_changed=a["support"] != b["support"],
                              same_q=qdelta == 0, same_shape=a["shape"] == b["shape"],
                              B_exact=bool(np.array_equal(coefficients[left], coefficients[right])),
                              left_rss=a["rss"], right_rss=b["rss"], delta_rss=b["rss"]-a["rss"],
                              delta_msic=b["msic"]-a["msic"], left_exact=a["exact"], right_exact=b["exact"],
                              left_mode_f1=a["mode_f1"], right_mode_f1=b["mode_f1"],
                              truth_shape_within_caps=a["truth_shape_within_caps"]))
    assert len(models) == 750 and len(pairs) == 500
    return plan, models, pairs, assets


def analyze(models, pairs):
    groups, summaries, associations, cases, layers = defaultdict(list), [], [], [], []
    for row in pairs:
        groups[(row["scenario"], row["phase"])].append(row)
    for (scene, phase), rows in groups.items():
        assert len(rows) == 50
        d = distribution([r["delta_pmse"] for r in rows])
        d.update(scenario=scene, phase=phase, support_changed=sum(r["support_changed"] for r in rows),
                 same_q_changed_support=sum(r["same_q"] and r["support_changed"] for r in rows),
                 same_shape_changed_support=sum(r["support_relation"] == "same_shape_different_support" for r in rows),
                 lower_rss_worse_pmse=sum(r["delta_rss"] < -1e-12 and r["delta_pmse"] > 1e-12 for r in rows),
                 same_shape_lower_rss_worse_pmse=sum(r["same_shape"] and r["delta_rss"] < -1e-12 and r["delta_pmse"] > 1e-12 for r in rows),
                 same_q_changed_support_worse=sum(r["same_q"] and r["support_changed"] and r["delta_pmse"] > 1e-12 for r in rows))
        summaries.append(d)
        for field in ("q_direction", "support_relation", "outcome"):
            for category in sorted({r[field] for r in rows}):
                group = [r for r in rows if r[field] == category]
                associations.append(dict(scenario=scene, phase=phase, grouping=field, category=category,
                                         n=len(group), mean_delta_pmse=float(np.mean([r["delta_pmse"] for r in group])),
                                         median_delta_pmse=float(np.median([r["delta_pmse"] for r in group])),
                                         mean_left_q=float(np.mean([r["left_q"] for r in group])),
                                         mean_right_q=float(np.mean([r["right_q"] for r in group])),
                                         improved=sum(r["delta_pmse"] < -1e-12 for r in group),
                                         tied=sum(abs(r["delta_pmse"]) <= 1e-12 for r in group),
                                         worsened=sum(r["delta_pmse"] > 1e-12 for r in group)))
        ordered = sorted(rows, key=lambda r: (r["delta_pmse"], r["task_id"]))
        rules = [("minimum_delta", ordered[0]), ("lower_median_delta", ordered[24]), ("maximum_delta", ordered[-1])]
        same_shape = [r for r in rows if r["same_shape"] and r["support_changed"] and r["delta_pmse"] > 1e-12]
        if same_shape:
            rules.append(("maximum_same_shape_changed_support_deterioration", max(same_shape, key=lambda r:(r["delta_pmse"], -r["task_id"]))))
        for rule, row in rules:
            cases.append(dict(selection_rule=rule, **row))
    for scene in sorted({r["scenario"] for r in models}):
        for version in ("v5", "v51", "v52"):
            rows = [r for r in models if r["scenario"] == scene and r["version"] == version]
            exact = sum(r["exact"] for r in rows)
            layers.append(dict(scenario=scene, version=version, n=50,
                               truth_q_legal=sum(r["true_q_within_Q"] for r in rows),
                               truth_caps_legal=sum(r["truth_shape_within_caps"] for r in rows),
                               truth_in_fixed_grid=sum(r["truth_shape_in_fixed_grid"] for r in rows),
                               selected_exact=exact, truth_pool_hit_proven_lower_bound=exact,
                               truth_pool_hit_other_cases="not available", true_pool_missed_count="not available",
                               oracle_selection_failure_count="not available", ols_rank_condition="not available",
                               q_mean=float(np.mean([r["q"] for r in rows])),
                               pmse_mean=float(np.mean([r["pmse"] for r in rows])),
                               pmse_median=float(np.median([r["pmse"] for r in rows])),
                               pmse_p90=float(np.quantile([r["pmse"] for r in rows], .9)),
                               pmse_p95=float(np.quantile([r["pmse"] for r in rows], .95))))
    return summaries, associations, cases, layers


def report_path(root, out):
    """A custom audit output never rewrites the current delivery report."""
    default_output = root / "doc/taskcard9/attribution"
    destination = root / "doc/taskcard9" if out.resolve() == default_output.resolve() else out
    return destination / "N100_REGRESSION_ATTRIBUTION.md"


def report(root, out, plan, models, pairs, summaries, associations, cases, layers):
    n100 = next(s for s in summaries if s["scenario"] == "n100" and s["phase"] == "BCD")
    n400 = next(s for s in summaries if s["scenario"] == "n400" and s["phase"] == "Grid")
    lines = ["# n100退化及n400对照：逐配对归因", "", "2026-10-17；research_only=true，release_approved=false。仅重分析冻结资产，新增拟合、数据生成、预测和merge均为0。", "",
             f"**结论：** n100的BCD阶段有{n100['worsened']}/50恶化、{n100['improved']}/50改善；恶化前5例占全部正向损失{100*n100['top5_positive_share_of_gross']:.2f}%，不是仅由一两个极端例造成。q增加与恶化集中伴随，但另有{n100['same_q_changed_support_worse']}例同q同形状变支撑后恶化，不能只归因于规模。n400网格阶段{n400['improved']}/50改善、无恶化，必须保留完整合法规模覆盖。Task7缺完整池与原X/y，尚不能把n100退化主要定性为MSIC漏选或OLS数值失稳。", "",
             "## 可识别的版本效应", "", "每场景50个同输入配对；ΔBCD=v5.1−v5，ΔGrid=v5.2−v5.1。PMSE差<0为改善，数值相同阈值1e−12。以下为指定设计的描述性版本效应，非对所有数据机制的因果普遍性保证。", "",
             "| 场景 | 阶段 | Δ均值 | Δ中位数 | ΔP90 | ΔP95 | 改善/同/恶化 | 支撑改变 | 同q变支撑恶化 |", "|---|---|---:|---:|---:|---:|---|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['scenario']} | {s['phase']} | {s['mean']:.6f} | {s['median']:.6f} | {s['p90']:.6f} | {s['p95']:.6f} | {s['improved']}/{s['tied']}/{s['worsened']} | {s['support_changed']} | {s['same_q_changed_support_worse']} |")
    lines += ["", "| 场景 | 版本 | PMSE均值 | 中位数 | P90 | P95 | q均值 | Exact |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for l in layers:
        if l["scenario"] in ("n100", "n400"):
            lines.append(f"| {l['scenario']} | {l['version']} | {l['pmse_mean']:.6f} | {l['pmse_median']:.6f} | {l['pmse_p90']:.6f} | {l['pmse_p95']:.6f} | {l['q_mean']:.2f} | {l['selected_exact']}/50 |")
    lines += ["", "## 尾部与q、支撑的关联", "", "gross-positive为全部正向PMSE差之和；net为改善与恶化抵消后的差之和。前5/10指按恶化差降序，若不足则取全部；占net可能超过100%，其含义是改善抵消了一部分损失，不能解释为概率。分位数为NumPy线性样本分位数，非置信区间。", "",
              "| 场景/阶段 | 正恶化和 | 净变化和 | 前5占正恶化 | 前10占正恶化 | 前5占净变化 | RSS降而PMSE升 | 同形状RSS降而PMSE升 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in summaries:
        pct = lambda x: "NA" if x is None else f"{100*x:.2f}%"
        lines.append(f"| {s['scenario']}/{s['phase']} | {s['gross_positive_sum']:.6f} | {s['net_sum']:.6f} | {pct(s['top5_positive_share_of_gross'])} | {pct(s['top10_positive_share_of_gross'])} | {pct(s['top5_positive_share_of_net'])} | {s['lower_rss_worse_pmse']} | {s['same_shape_lower_rss_worse_pmse']} |")
    lines += ["", "| 场景/阶段 | q变化 | N | PMSE改善/同/恶化 | 平均ΔPMSE | 旧q均值→新q均值 |", "|---|---|---:|---|---:|---|"]
    for a in associations:
        if a["grouping"] == "q_direction" and a["scenario"] in ("n100", "n400"):
            lines.append(f"| {a['scenario']}/{a['phase']} | {a['category']} | {a['n']} | {a['improved']}/{a['tied']}/{a['worsened']} | {a['mean_delta_pmse']:.6f} | {a['mean_left_q']:.2f}→{a['mean_right_q']:.2f} |")
    lines += ["", "q、形状及支撑变化是版本修改后的共同结果；以上关联不把q变化独立识别成因果中介，也不证明OLS数值失稳。所有750选中模型的REE/Exact/F1从保存B/真支撑可复核；原X/y未归档，所以不能独立复算RSS/PMSE，也不能由q、REE或RSS推算rank/condition。", "", "## 五层归因", "", "| 场景 | 版本 | 真q合法 | 真形状caps合法 | 真形状进入fixed网格 | 最终Exact | 真位置曾入完整池 | 选择失败 | OLS秩/条件数 |", "|---|---|---:|---:|---:|---:|---|---|---|"]
    for l in layers:
        if l["scenario"] in ("n100", "n400"):
            lines.append(f"| {l['scenario']} | {l['version']} | {l['truth_q_legal']}/50 | {l['truth_caps_legal']}/50 | {l['truth_in_fixed_grid']}/50 | {l['selected_exact']}/50 | 至少{l['selected_exact']}；其余not available | not available | not available |")
    lines += ["", "1. n100的真形状(4,3,5)、q=60≤Q=70，但caps=(4,4,4)使50/50不可表示；这是结构性恢复障碍，不能要求现有选择器恢复真形状。该指标在n100内没有变异，不能估计其与预测恶化的组内相关或独立因果效应；n400也不是只改caps的匹配实验。", "2. fixed规模网格与额外adaptive终态必须分开：n400旧fixed网格不含真形状，旧最终仍有1个Exact，表明adaptive候选可以找到；不能宣称旧完整候选池无真模型。v5.2合法fixed规模补全保留其恢复机会。", "3. 全池候选未保存，因此最终未恢复不能被写成位置从未找到；也不能判断是否有真位置/更好测试候选被MSIC错过。Exact只提供“曾入池”的正向证据下界，不提供未入池的反向证据。", "4. RSS降低且PMSE升高支持训练拟合与泛化偏离的现象；同形状时复杂度项完全相同，只改依赖q/形状的惩罚不能区分这两类支撑。此处是跨版本所选模型比较，不是同一个冻结候选池的候选对照，不能据此计算pool oracle。", "", "## 机械规则代表案例", "", "案例只用于事后解释，不作新的效果估计或选参；每场景/阶段选Δ最小、下中位第25、Δ最大（task_id确定并列）；若存在同形状变支撑恶化，再选其中Δ最大。完整支撑见逐对CSV。", "", "| 场景/阶段 | 规则 | Task | q与形状 | PMSE旧→新 | ΔRSS | 支撑关系 |", "|---|---|---:|---|---|---:|---|"]
    for c in cases:
        if c["scenario"] in ("n100", "n400"):
            lines.append(f"| {c['scenario']}/{c['phase']} | {c['selection_rule']} | {c['task_id']} | {c['left_q']} {c['left_shape']}→{c['right_q']} {c['right_shape']} | {c['left_pmse']:.6f}→{c['right_pmse']:.6f} | {c['delta_rss']:.6f} | {c['support_relation']} |")
    lines += ["", "## 证据与限制", "", "250输入身份唯一，但仅150个(seed,test_seed)随机族，部分场景共用随机种子，禁止合并为250个IID重复或把跨场景差当单因素因果效应。主分析n100/n400，弱SNR/rho0为辅助场景，均分开报告；没有进行显著性检验或事后调整门槛。", "", "[750逐模型结果](attribution/selected_models.csv)、[500逐对阶段变化](attribution/paired_changes.csv)、[分布统计](attribution/effect_distribution.csv)、[关联表](attribution/associations.csv)、[代表案例](attribution/representative_cases.csv)、[五层表及各版本PMSE尾部](attribution/layer_attribution.csv)、[核验记录](attribution/validation.json)。", "", "复现：仓库根目录运行 `.venv/Scripts/python.exe -B research/taskcard9/attribution.py`；当前输出拒绝覆盖，须先把输出参数指向新的审计目录（`--output`）。不运行任何历史main，不重新merge。"]
    body = "\n".join(lines) + "\n"
    destination = report_path(root, out)
    if destination.parent.resolve() == out.resolve():
        body = body.replace("](attribution/", "](")
    destination.write_text(body, encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="doc/taskcard9/attribution")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), "Choose a fresh output directory; never overwrite prior evidence"
    cluster_audit = load_audit(root / "doc/taskcard7_completion/audit_cluster.py", "cluster_audit", root)
    estimate_audit = load_audit(root / "doc/taskcard7_completion/audit_estimates.py", "estimate_audit", root)
    plan, models, pairs, assets = collect(root)
    summaries, associations, cases, layers = analyze(models, pairs)
    families = Counter((t["seed"], t["test_seed"]) for t in plan["tasks"])
    assert len(families) == 150 and len({t["input_key"] for t in plan["tasks"]}) == 250
    assert all(r["B_exact"] for r in pairs if r["phase"] == "Grid" and r["scenario"] != "n400")
    validation = dict(status="PASS", tasks=250, selected_models=750, paired_phase_records=500,
                      unique_inputs=250, seed_families=150, family_multiplicity=dict(Counter(families.values())),
                      no_fits=True, no_generation=True, no_predictions=True, no_merge=True,
                      original_cluster_identity_audit=cluster_audit, independent_saved_metric_audit=estimate_audit,
                      readonly_original_audit_sources={p:sha(root/p) for p in ("doc/taskcard7_completion/audit_cluster.py", "doc/taskcard7_completion/audit_estimates.py")},
                      runtime=dict(python=__import__("sys").version, numpy=np.__version__),
                      verification_scope="PMSE/RSS remain sealed-record metrics; no raw X/y or test streams and no candidate pool archived in these 250 tasks.")
    for name, rows in (("selected_models",models), ("paired_changes",pairs), ("effect_distribution",summaries),
                       ("associations",associations), ("representative_cases",cases), ("layer_attribution",layers)):
        save_csv(out / (name + ".csv"), rows)
    save_json(out / "summary.json", dict(effects=summaries, layers=layers, seed_families=150, unique_inputs=250))
    save_json(out / "validation.json", validation)
    save_json(out / "source_assets.json", assets)
    report(root, out, plan, models, pairs, summaries, associations, cases, layers)
    print(json.dumps(dict(status="PASS", model_count=len(models), pair_count=len(pairs),
                         key_summaries=[s for s in summaries if s["scenario"] in ("n100", "n400")]), ensure_ascii=False))


if __name__ == "__main__":
    main()
