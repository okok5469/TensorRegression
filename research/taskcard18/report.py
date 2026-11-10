"""Read existing evidence; --write refreshes three derived closeout reports.

This module never fits, reconstructs inputs, evaluates test samples, merges
historical runs, or launches jobs. Missing work is pending, not a failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo


def derive_status(facts: dict) -> str:
    if facts.get("blocking_issues"):
        return "BLOCKED"
    gates = facts.get("gates", {})
    passed = all(gates.get(f"G{i}", {}).get("status") in {"PASS", "PASS_CURRENT"} for i in range(8))
    complete = facts.get("valid_fits", 0) == 1200 and facts.get("evaluations", 0) == 1200
    return "FINAL" if complete and passed else "IN_PROGRESS"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else None


def _csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _verify_seal(value: dict) -> None:
    if value.get("record_sha256") != _digest({k: v for k, v in value.items() if k != "record_sha256"}):
        raise ValueError("Immutable record seal mismatch")


def collect_snapshot(root: Path) -> dict:
    """Validate completed records only; do not reopen all 700 historical NPZs."""
    import numpy as np
    from .accept import validate_arrays, validate_metadata
    from .identity import verify_hashes

    doc = root / "doc/taskcard18"
    output = root / "numerical_simulation/outputs_taskcard18_closeout"
    historical = root / "numerical_simulation/outputs_v52_search_final_full"
    issues, fit_rows, evaluated, failures = [], [], [], []
    audit = _json(doc / "IDENTITY_AUDIT.json") or {}
    plan = _json(historical / "plan.json") or {}
    manifest = _json(root / "doc/releases/v5.2-search-final/RELEASE_MANIFEST.json") or {}
    protocol = _json(doc / "CLOSEOUT_PROTOCOL.json") or {}
    amendment = _json(doc / "EXECUTION_AMENDMENT.json") or {}
    cluster_execution = amendment.get("execution") == "existing_cluster" and amendment.get("user_authorized") is True
    if cluster_execution and (amendment.get("statistics_unchanged") is not True or amendment.get("original_protocol_sha256") != _sha(doc / "CLOSEOUT_PROTOCOL.json")):
        issues.append("Execution amendment is not bound to the unchanged statistical protocol")
    mapping = _csv(doc / "input_identity_map.csv")
    by_task = {int(row["task_id"]): row for row in mapping}
    expected_tasks = {int(t["task_id"]): t for t in plan.get("tasks", [])}
    unique_inputs = {row["unique_input_id"] for row in mapping}
    config_counts = Counter()
    for config in {row["config_id"] for row in mapping}:
        config_counts[config] = len({row["unique_input_id"] for row in mapping if row["config_id"] == config})
    production_pass = simulation_pass = mapping_pass = run_pass = False
    try:
        sources = manifest["source_sha256"]
        initial = audit.get("production", {})
        if initial.get("source_sha256") != sources or initial.get("manifest_sha256") != _sha(root / "doc/releases/v5.2-search-final/RELEASE_MANIFEST.json"):
            raise ValueError("Release identity differs from the original Task18 audit")
        verify_hashes(root, sources)
        inventory = {p.relative_to(root).as_posix() for p in (root / "src/tensor_abess").rglob("*.py")}
        if len(sources) != 32 or inventory != set(sources):
            raise ValueError("Production inventory differs from the 32-source release")
        spec = importlib.util.find_spec("tensor_abess")
        if spec is None or Path(spec.origin).resolve() != (root / "src/tensor_abess/__init__.py").resolve():
            raise ValueError("Actual tensor_abess import path differs from this repository")
        production_pass = True
        verify_hashes(root, plan["source_fingerprints"])
        if len(plan["source_fingerprints"]) != 38 or plan["plan_id"] != _digest({k: v for k, v in plan.items() if k != "plan_id"}):
            raise ValueError("Frozen simulation plan seal/source inventory conflict")
        simulation_pass = True
    except (ValueError, KeyError, OSError) as error:
        issues.append("Frozen production/plan identity: " + str(error))
    run = _json(output / "RUN_IDENTITY.json")
    try:
        if len(mapping) != 700 or len(by_task) != 700 or len(unique_inputs) != 600 or sorted(config_counts.values()) != [50] * 9 + [150]:
            raise ValueError("Expected 700 labels, 600 complete inputs and 10 configuration denominators")
        for row in mapping:
            task = expected_tasks[int(row["task_id"])]
            fingerprints = {key: row[key + "_sha256"] for key in ("X_train", "y_train", "B_true", "train_noise")}
            if row["unique_input_id"] != "input_" + _digest({"input_fingerprints": fingerprints, "test_seed": task["test_seed"]}):
                raise ValueError("Input-map canonical complete identity conflict")
            if int(row["train_seed"]) != task["seed"] or int(row["test_seed"]) != task["test_seed"]:
                raise ValueError("Input-map seed conflict")
        mapping_pass = True
        if run is not None:
            _verify_seal(run)
            if run["plan_id"] != plan["plan_id"] or run["plan_file_sha256"] != _sha(historical / "plan.json") or run["input_map_sha256"] != _sha(doc / "input_identity_map.csv"):
                raise ValueError("Run identity differs from frozen plan/input map")
            verify_hashes(root, {"research/taskcard18/" + name: value for name, value in run["identity"]["adapter_hashes"].items()})
            if run["identity"]["protocol_sha256"] != _sha(doc / "CLOSEOUT_PROTOCOL.json"):
                raise ValueError("Run protocol identity conflict")
            for task in run["tasks"]:
                old = expected_tasks[task["task_id"]]
                if task["unique_input_id"] != by_task[task["task_id"]]["unique_input_id"] or task["config_sha256"] != _digest(old["config"]) or task["seed"] != old["seed"] or task["test_seed"] != old["test_seed"]:
                    raise ValueError("Run task identity conflict")
            if len(run["tasks"]) != 700:
                raise ValueError("Incomplete immutable run task inventory")
            run_pass = True
    except (ValueError, KeyError, OSError) as error:
        issues.append("Input/run identity: " + str(error))
    expected_identity = run["identity"] if run else None
    accepted = {}
    for path in sorted((output / "fits").glob("*.json")):
        try:
            metadata = _json(path)
            method = path.stem.rsplit("_", 1)[-1]
            _verify_seal(metadata)
            task = expected_tasks[metadata["task_id"]]
            uid = by_task[task["task_id"]]["unique_input_id"]
            key = (uid, method)
            if key in accepted or path.stem != uid + "_" + method or metadata["unique_input_id"] != uid or metadata["identity"] != expected_identity:
                raise ValueError("Completed fit input/method/execution identity conflict")
            if metadata["npz_sha256"] != _sha(path.with_suffix(".npz")):
                raise ValueError("Completed coefficient-file hash conflict")
            validate_metadata(metadata, method)
            for d in metadata["diagnostics"]:
                n = task["config"]["n_train"]
                expected_n = n if d["fit_kind"] == "final" else n - (n // 5 + (d["fold_index"] < n % 5))
                if d["n_fit"] != expected_n:
                    raise ValueError("Certificate sample-count conflict")
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                validate_arrays(arrays["B"], [arrays[f"S{i}"] for i in (1, 2, 3)], task["config"]["dims"])
            correspondence = metadata.get("input_correspondence", {})
            if correspondence.get("valid") is not True or correspondence.get("unique_input_id") != uid:
                raise ValueError("Completed fit lacks matching input-correspondence evidence")
            accepted[key] = metadata
            fit_rows.append({"task_id": task["task_id"], "unique_input_id": uid, "method": method,
                             "runtime_sec": metadata["runtime_sec"], "certificates": len(metadata["diagnostics"]),
                             "max_kkt_ratio": max(d["kkt_ratio"] or 0.0 for d in metadata["diagnostics"]),
                             "max_normalized_gap": max(d["gap_normalized"] or 0.0 for d in metadata["diagnostics"])})
        except (ValueError, KeyError, TypeError, OSError) as error:
            issues.append(path.relative_to(root).as_posix() + ": " + str(error))
    for path in sorted((output / "failures").glob("*.json")):
        failure = _json(path)
        resolved = (failure.get("unique_input_id"), failure.get("method")) in accepted
        failures.append({"path": path.relative_to(root).as_posix(), "resolved_by_valid_fit": resolved,
                         "task_id": failure.get("task_id"), "method": failure.get("method"), "error": failure.get("error")})
        if not resolved:
            issues.append("Unresolved immutable fit failure: " + path.relative_to(root).as_posix())
    for path in sorted((output / "evaluations").glob("*.json")):
        try:
            evaluation = _json(path)
            method = path.stem.rsplit("_", 1)[-1]
            key = (evaluation["unique_input_id"], method)
            if path.stem != key[0] + "_" + method or key in evaluated:
                raise ValueError("Duplicate/noncanonical evaluation identity")
            fitted = accepted[key]
            if evaluation["identity"] != expected_identity or evaluation["fit_npz_sha256"] != fitted["npz_sha256"] or evaluation["method_id"] != fitted["method_id"] or len(evaluation["test_stream_hashes"]) != 5:
                raise ValueError("Evaluation identity/model/test-stream conflict")
            if not all(math.isfinite(evaluation["metrics"][name]) for name in ("pmse", "signal_mse", "r2_test")):
                raise ValueError("Nonfinite evaluation")
            evaluated.append(key)
        except (ValueError, KeyError, TypeError, OSError) as error:
            issues.append(path.relative_to(root).as_posix() + ": " + str(error))
    archived = []
    for path in sorted((output / "initial_training_diagnostics/fits").glob("*.json")):
        saved = _json(path)
        archived.append({"method_id": saved.get("method_id"), "runtime_sec": saved.get("runtime_sec"),
                         "certificates": len(saved.get("diagnostics", [])), "test_evaluated": saved.get("test_evaluated", False)})
    input_pilot = _json(output / "pilot_input_audit_v2.json") or {}
    verified_inputs = {item["unique_input_id"] for item in input_pilot.get("checks", []) if item.get("valid")}
    verified_inputs.update(uid for uid, _ in accepted)
    pilot_uids = {by_task[t["task_id"]]["unique_input_id"] for t in expected_tasks.values()
                  if t["scenario_id"] in protocol.get("pilot_scenarios", []) and t["replicate"] in protocol.get("pilot_replicates", [])}
    pilot_count = sum(uid in pilot_uids for uid, _ in accepted)
    pilot_gate = _json(output / "PILOT_GATE.json") or {}
    pilot_pass = pilot_gate.get("status") == "PASS" and pilot_gate.get("identity") == expected_identity and pilot_gate.get("fits") == 40
    analysis_counts = {name: len(_csv(doc / "results" / f"certified_{name}.csv")) for name in
                       ("label_metrics", "unique_metrics", "config_summary", "14_label_summary", "paired_pmse")}
    analysis_complete = analysis_counts == {"label_metrics": 2800, "unique_metrics": 2400, "config_summary": 40, "14_label_summary": 56, "paired_pmse": 30}
    pairs = _csv(doc / "results/certified_paired_pmse.csv")
    expected_pairs = {(config, "adaptive_minus_" + method) for config in config_counts for method in ("lasso", "glasso", "oracle")}
    paired_valid = len(pairs) == 30 and {(r["config_id"], r["comparison"]) for r in pairs} == expected_pairs and all(int(row["n"]) == config_counts[row["config_id"]]
                    and int(row["df"]) == int(row["n"]) - 1
                    and all(math.isfinite(float(row[k])) for k in ("mean_difference", "mcse", "ci_low", "ci_high")) for row in pairs)
    figure_qa = _json(doc / "figure_qa/FIGURE_VALIDATION.json") or {}
    final_figures = bool(figure_qa.get("status") == "PASS" and figure_qa.get("not_complete_revised_baseline_comparison") is False)
    review_path = doc / "INDEPENDENT_REVIEW.md"
    final_review = review_path.read_text(encoding="utf-8") if review_path.exists() else ""
    final_review_pass = bool(re.search(r"(?im)^\s*(?:status|状态)\s*[:：]\s*(?:\*\*)?PASS(?:\*\*)?\s*$", final_review))
    complete = len(accepted) == 1200 and len(evaluated) == 1200
    final_review_sources = [root / name for name in ("paper/sections/02_method.md", "paper/sections/04_simulation.md", "paper/appendix/simulation_protocol.md", "doc/taskcard18/figure_qa/FIGURE_VALIDATION.json", "doc/taskcard18/results/certified_config_summary.csv", "doc/taskcard18/results/certified_paired_pmse.csv")]
    final_review_bound = all(p.exists() and _sha(p) in final_review for p in final_review_sources)
    def gate(status, evidence, remaining):
        return {"status": status, "evidence": evidence, "remaining": "已验收" if status == "PASS" else remaining}
    gates = {
        "G0": gate("PASS_CURRENT" if production_pass and simulation_pass else "BLOCKED", ["IDENTITY_AUDIT.json", "release manifest", "current32/38SHA256"], "结束时仍须再次复验；当前检查不重跑700任务"),
        "G1": gate("PASS" if mapping_pass and len(verified_inputs) == 600 else "PARTIAL", ["input_identity_map.csv", "pilot_input_audit_v2.json", "fit input_correspondence"], "核实全部600输入的数值对应；不声称字节身份"),
        "G2": gate("PASS" if len(accepted) == 1200 else "PARTIAL", ["CLOSEOUT_PROTOCOL.json", "completed fit certificates"], "全部1200新方法拟合的折边界/CV选择需通过"),
        "G3": gate("PASS" if len(accepted) == 1200 and pilot_pass else "PENDING", ["fits/", "PILOT_GATE.json"], f"当前pilot {pilot_count}/40；pilot后仍需600×2全量证书；不得仅pilot结束"),
        "G4": gate("PASS" if complete and analysis_complete else "PARTIAL", ["METRICS_AND_STATISTICS.md", "results/"], "补齐新可靠对照保存系数及实际元素/声明/投影/容量重算"),
        "G5": gate("PASS" if complete and analysis_complete and paired_valid else "PARTIAL", ["results/historical_paired_pmse.csv", "results/certified_paired_pmse.csv"], "全部同输入配对结果与有效分母、CI仍需完成"),
        "G6": gate("PASS" if run_pass and complete and analysis_complete and final_figures and final_review_bound else "PARTIAL", ["RUN_IDENTITY.json", "fit/evaluation seals", "figure QA and review hashes"], "完整拟合→评价→统计→图表身份链及独立验收待完成"),
        "G7": gate("PASS" if complete and final_figures and final_review_pass and final_review_bound else "PENDING", ["MANUSCRIPT_INDEPENDENT_REVIEW.md", "paper/sections/", "INDEPENDENT_REVIEW.md"], "方法/已核实设定通过有限范围审查；全新比较正文/表图及最终独立审查未通过"),
    }
    legacy = [row for row in _csv(doc / "results/historical_unique_metrics.csv") if row["method"] == "lasso"]
    history = {"legacy_lasso_actual_nnz_mean": sum(float(r["element_nonzero"]) for r in legacy) / len(legacy) if legacy else None,
               "legacy_lasso_q_rect_mean": sum(float(r["q_rect"]) for r in legacy) / len(legacy) if legacy else None}
    summary = _csv(doc / "results/historical_config_summary.csv")
    baseline_cfg = next((config for config, n in config_counts.items() if n == 150), None)
    baseline = next((r for r in summary if r["method"] == "adaptive" and r["config_id"] == baseline_cfg), {})
    n400_cfg = next((by_task[t["task_id"]]["config_id"] for t in expected_tasks.values() if t["config"]["n_train"] == 400), None)
    n400 = next((r for r in summary if r["method"] == "adaptive" and r["config_id"] == n400_cfg), {})
    history.update({"baseline_n": int(baseline.get("n", 0)), "baseline_pmse": baseline.get("pmse_mean"),
                    "baseline_exact": baseline.get("exact_successes"), "n400_n": int(n400.get("n", 0)), "n400_exact": n400.get("exact_successes")})
    facts = {"schema_version": 1, "checked_beijing": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
             "valid_fits": len(accepted), "evaluations": len(evaluated), "blocking_issues": issues, "gates": gates,
             "chapter_status": "DRAFT", "execution": "existing_cluster" if cluster_execution else "local_existing_environment_only",
             "launcher_session_reference": "cluster_submission.json" if cluster_execution else "68026",
             "execution_note": ("用户已停止本机并授权集群交付；本快照不声称已提交集群。" if cluster_execution and not (output / "cluster_submission.json").exists() else ("仅按本包cluster_submission.json记录定位集群作业，进度以实际缓存为准。" if cluster_execution else "执行限定已有本机环境，不使用集群。")),
             "activity_note": "只由完成记录计数；不把session引用当进程存活或任务完成证明", "planned_unique_inputs": 600,
             "planned_new_method_fits": 1200, "historical_labels": len(mapping), "unique_inputs": len(unique_inputs),
             "configurations": len(config_counts), "configuration_unique_counts": dict(config_counts),
             "reusable_frozen_unique_method_results": 1200 if audit.get("status") == "IDENTITY_PASS" and mapping_pass else 0,
             "fits_by_method": dict(Counter(row["method"] for row in fit_rows)),
             "fitted_unique_inputs": len({uid for uid, _ in accepted}), "inputs_with_both_fits": sum((uid, "lasso") in accepted and (uid, "glasso") in accepted for uid in unique_inputs),
             "verified_input_correspondences": len(verified_inputs), "pilot_valid_fits": pilot_count,
             "pilot_gate_status": pilot_gate.get("status", "PENDING"), "fit_records": fit_rows, "failure_records": failures,
             "archived_v0_training_only": archived, "analysis_row_counts": analysis_counts,
             "figure_status": figure_qa.get("status", "PENDING"), "history": history,
             "paired_pmse": pairs if complete and analysis_complete and paired_valid else [],
             "training_protocol_revisions_used": protocol.get("training_protocol_revisions_used", 0),
             "new_method_ids": protocol.get("new_method_ids", [])}
    facts["status"] = derive_status(facts)
    facts["chapter_status"] = "FINAL" if facts["status"] == "FINAL" else "DRAFT"
    return facts


def render_report(facts: dict) -> str:
    h = facts["history"]
    final = facts["status"] == "FINAL"
    method_counts = "、".join(f"{method} {count}" for method, count in sorted(facts["fits_by_method"].items())) or "尚无"
    chapter_note = "联合定稿条件已满足。" if final else "尚未联合定稿。"
    remaining_note = "本卡范围内有效性验收已完成。" if final else "当前范围内仍须完成两个主对照的全部有效比较与章节联合验收。"
    figure_note = "完整新对照图表已验收。" if final else ("目前四类图及表只包含已核实封存ABESS/Oracle，不冒充新完整比较。" if facts["figure_status"] == "PASS_FROZEN_METHOD_DRAFT_FIGURES" else "完整新对照主图表仍待全部验收。")
    comparison_note = "当前不作方向性判断；全量有效对照、评价与同输入配对尚未完成，不能从CV、单个pilot或旧排名判断优势保留/缩小/消失/反转。"
    if facts["paired_pmse"]:
        outcomes = []
        for row in facts["paired_pmse"]:
            outcome = "ABESS的PMSE较低" if float(row["ci_high"]) < 0 else ("ABESS的PMSE较高" if float(row["ci_low"]) > 0 else "差异方向尚不明确")
            outcomes.append(f"{row['config_id']} / {row['comparison']}: 配对差={row['mean_difference']}, CI=[{row['ci_low']},{row['ci_high']}], {outcome}")
        comparison_note = "同输入完整配对结果：" + "；".join(outcomes) + "。上述区间逐配置解释，不累计显著获胜次数。"
    problems = "\n".join("- " + item for item in facts["blocking_issues"]) or "无已确认未解关键失败；尚缺的运行结果属于进行中的工作。"
    gates = "\n".join(f"| {key} | {value['status']} | {value['remaining']} |" for key, value in facts["gates"].items())
    completed = "\n".join(f"| {r['task_id']} | {r['method']} | {r['certificates']} | {r['runtime_sec']:.2f} |" for r in facts["fit_records"]) or "| — | 尚无完整验收的新拟合 | — | — |"
    return f"""# 任务卡18动态执行与验收报告

生成时间（北京时间）：{facts['checked_beijing']}。**任务状态：{facts['status']}；两章状态：{facts['chapter_status']}。{chapter_note}** 本文件由只读证据入口生成，文件名含FINAL不表示任务已完成；这是动态状态快照。

1. **方法章和模拟章是否定稿？** 当前为{facts['chapter_status']}；新对照有效拟合{facts['valid_fits']}/1200，新增方法测试评价{facts['evaluations']}/1200。固定训练pilot已验{facts['pilot_valid_fits']}/40，pilot Gate={facts['pilot_gate_status']}。执行继续按pilot→全600×2→评价→统计表图→独立验收推进，**任务不得仅在pilot结束**。
2. **正式算法是否冻结？** G0={facts['gates']['G0']['status']}，本次重读release并检查32生产源、38模拟源码/配置及原计划seal，实际import路径也核对；没有重新拟合正式ABESS。结束时仍须再次检查。{facts['execution_note']}运行定位引用{facts['launcher_session_reference']}不代替存活或完成证据。
3. **基线问题及修复是什么？** 旧Vector Lasso在同lambda内跨CV折传beta，且两旧对照均缺全部必要凸拟合证书。隔离适配器保留无截距/无标准化、原20点网格、5折SSE与并列规则；每折状态隔离、最终全训练独立拟合，Lasso保留L1，顺序Tensor-GLasso保留预筛、顺序、非空回退、局部CV和矩形OLS。v2仅用每至多50扫检查Lasso双证书提前停止，硬阈值与2000/10000/50000累计预算不变；Group数值流程不变。当前method_id为{', '.join(facts['new_method_ids'])}，唯一训练协议修订已使用{facts['training_protocol_revisions_used']}次。
4. **本轮实际补算/复用多少？** v2完成并验收{facts['valid_fits']}个新方法拟合，覆盖{facts['fitted_unique_inputs']}个唯一输入，两个方法均齐的输入{facts['inputs_with_both_fits']}个；方法计数：{method_counts}。输入数值对应已核{facts['verified_input_correspondences']}/600，不能声称原未保存数组逐字节相同。复用已验正式ABESS与Oracle共{facts['reusable_frozen_unique_method_results']}个唯一方法结果。旧700标签按600完整输入/10配置映射；基准150，其余各50。初版v0仅有{len(facts['archived_v0_training_only'])}个封存训练诊断，其证书与耗时见gates.json，**不计v2主比较**。
5. **更新对照后哪些优势改变？** {comparison_note}旧“11/14胜出”不作为本轮规范比较证据。
6. **支撑与不确定性有什么新解释？** 历史旧Lasso600唯一输入描述均值：实际非零{h['legacy_lasso_actual_nnz_mean']:.2f}，矩形容量{h['legacy_lasso_q_rect_mean']:.2f}，二者不可互换；这些是旧未认证实现。正式ABESS基准150输入PMSE={h['baseline_pmse']}、Exact={h['baseline_exact']}/{h['baseline_n']}；n400为{h['n400_exact']}/{h['n400_n']}。这些属于封存正式ABESS描述，不是新对照优势结论。SD、MCSE、t均值区间、Wilson恢复区间分列，同一完整输入配对，禁止池化600/700为IID显著性。n100还须解释caps(4,4,4)排除真形状(4,3,5)。
7. **交付物在哪里？** 正文：paper/sections/02_method.md、04_simulation.md；附录：paper/appendix/simulation_protocol.md；配置/统计缓存：doc/taskcard18/results/；{figure_note}QA={facts['figure_status']}。方法/已核实设定审查：MANUSCRIPT_INDEPENDENT_REVIEW.md。复现入口：research/taskcard18/README.md。动态状态：本报告、gates.json与paper/CHAPTER_FREEZE.md；均不触发拟合、重建输入、评价或作业提交。
8. **哪些后续不属于本卡？** 全局最优/一致性/最优率等理论、新实证数据拟合、方法新颖性与整篇论文贡献定位不在本卡收尾范围；本卡也不证明整篇论文已达投稿条件。{remaining_note}不能用FINAL_WITH_LIMITATIONS豁免G2/G3。

## 实际已完成的v2训练拟合

| task_id | 方法 | 必要凸拟合证书数 | 秒 |
|---|---|---:|---:|
{completed}

## Gate快照

| Gate | 状态 | 剩余验收范围 |
|---|---|---|
{gates}

## 关键未解失败

{problems}

缺失的未完成结果不自动判BLOCKED。只有实际数值失败、身份冲突等未解关键有效性缺口才判BLOCKED；全1200拟合与1200评价且G0–G7有效性全部通过才可能FINAL。
"""


def write_reports(root: Path, facts: dict) -> None:
    doc = root / "doc/taskcard18"
    doc.mkdir(parents=True, exist_ok=True)
    (doc / "gates.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (doc / "TASKCARD18_FINAL_REPORT.md").write_text(render_report(facts), encoding="utf-8")
    final_note = "已满足联合定稿验收条件。" if facts["status"] == "FINAL" else "尚未联合定稿。"
    review_note = "完整比较、表图与最终独立验收均已满足Gate要求。" if facts["status"] == "FINAL" else "方法定义与已核实模拟设定已经过第二轮文稿审查，但该有限范围PASS不是独立基线验收，不授予G3或整体G7通过。当前图表状态为" + facts["figure_status"] + "，完整新对照主比较仍需验收。"
    remaining_note = "本卡范围内验收已经完成。" if facts["status"] == "FINAL" else "本卡仍须完成全600唯一输入×2对照、证书/评价/配对表图、正文更新及最终独立验收。"
    text = f"""# 方法章与模拟章冻结状态

更新时间（北京时间）：{facts['checked_beijing']}。

**章节状态：{facts['chapter_status']}；任务状态：{facts['status']}。** {final_note}动态报告入口不会因已存在文稿或完成pilot而自动结束任务。

正式32源与原MSIC配置保持冻结，当前G0={facts['gates']['G0']['status']}。新可靠对照已验{facts['valid_fits']}/1200个拟合，测试评价{facts['evaluations']}/1200；训练pilot{facts['pilot_valid_fits']}/40。{facts['execution_note']}保留原记录、全部失败及初版训练诊断。

{review_note}

{remaining_note}理论、新实证与整篇论文贡献定位另属后续研究。全部有效性Gate通过前，不称FINAL或FINAL_WITH_LIMITATIONS。

完整计数、证据及剩余事项见[动态执行报告](../doc/taskcard18/TASKCARD18_FINAL_REPORT.md)和[Gate快照](../doc/taskcard18/gates.json)。
"""
    (root / "paper/CHAPTER_FREEZE.md").write_text(text, encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh only the three derived status reports")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    facts = collect_snapshot(root)
    if args.write:
        write_reports(root, facts)
    print(json.dumps({k: facts[k] for k in ("status", "chapter_status", "valid_fits", "evaluations", "pilot_valid_fits", "blocking_issues")}, ensure_ascii=False, allow_nan=False))
    return 1 if facts["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
