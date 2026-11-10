"""Build certified four-method figures from accepted caches; no fitting/testing.

Python/matplotlib is the fixed backend. --frozen reproduces the older two-method
drafts; the default requires all40 certified configuration/method cells.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import norm, t


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wilson(successes, size):
    z = float(norm.ppf(0.975))
    p = successes / size
    denominator = 1 + z*z/size
    center = (p + z*z/(2*size))/denominator
    half = z*np.sqrt(p*(1-p)/size + z*z/(4*size*size))/denominator
    # Analytic binomial boundaries avoid round-off reversing an errorbar.
    return (0.0 if successes == 0 else center-half,
            1.0 if successes == size else center+half)


def verified_data(root):
    source_names = ["doc/taskcard18/results/historical_config_summary.csv",
                    "doc/taskcard18/input_identity_map.csv",
                    "numerical_simulation/outputs_v52_search_final_full/plan.json",
                    "numerical_simulation/outputs_v52_search_final_full/raw_results.csv"]
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_names}
    summary = read_csv(root/source_names[0])
    mapping = read_csv(root/source_names[1])
    plan = json.loads((root/source_names[2]).read_text(encoding="utf-8"))
    raw = read_csv(root/source_names[3])
    configurations, input_groups = {}, defaultdict(list)
    for row in mapping:
        task = plan["tasks"][int(row["task_id"])]
        cfg = {key: value for key, value in task["config"].items() if key != "random_state"}
        cfg_id = "cfg_" + hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False,
                                                  allow_nan=False).encode("utf-8")).hexdigest()
        if cfg_id != row["config_id"]:
            raise ValueError("Configuration ID differs from frozen plan")
        if (int(row["train_seed"]), int(row["test_seed"])) != (task["seed"], task["test_seed"]):
            raise ValueError("Seed mapping differs from frozen plan")
        configurations[cfg_id] = cfg
        input_groups[row["unique_input_id"]].append(row)
    if len(mapping) != 700 or len(input_groups) != 600 or len(configurations) != 10:
        raise ValueError("Unexpected frozen input/label/configuration counts")
    canonical = {min(int(row["task_id"]) for row in rows): rows[0]["config_id"]
                 for rows in input_groups.values()}
    if any(len({row["config_id"] for row in rows}) != 1 for rows in input_groups.values()):
        raise ValueError("An input alias crosses parameter configurations")
    grouped = defaultdict(list)
    for row in raw:
        task_id = int(row["task_id"])
        if task_id in canonical and row["method"] in {"adaptive", "oracle"}:
            grouped[(canonical[task_id], row["method"])].append(row)
    selected = [row for row in summary if row["method"] in {"adaptive", "oracle"}]
    if len(selected) != 20 or len(grouped) != 20:
        raise ValueError("Expected all10 configurations for both frozen methods")
    table = []
    checks = []
    display = {"adaptive": "Tensor-ABESS v5.2", "oracle": "Oracle OLS"}
    for row in selected:
        cfg = configurations[row["config_id"]]
        members = grouped[(row["config_id"], row["method"])]
        values = np.asarray([float(member["pmse"]) for member in members])
        successes = int(sum(float(member["exact_recovery"]) for member in members))
        size = len(members)
        if not np.isfinite(values).all() or size != int(row["n"]) or int(row["pmse_valid_n"]) != size:
            raise ValueError("Invalid/missing denominator in frozen PMSE")
        if int(row["exact_recovery_valid_n"]) != size or row["evidence_stage"] != "frozen_verified":
            raise ValueError("Exact denominator or evidence stage differs from frozen contract")
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        exact_low, exact_high = wilson(successes, size)
        observed = [mean, sd, successes, successes/size, exact_low, exact_high]
        saved = [float(row[key]) for key in ("pmse_mean", "pmse_sd", "exact_successes",
                                             "exact_recovery_mean", "exact_wilson_low", "exact_wilson_high")]
        np.testing.assert_allclose(observed, saved, rtol=1e-11, atol=1e-10)
        half = float(t.ppf(0.975, size-1))*sd/np.sqrt(size)
        table.append({"config_id": row["config_id"], "method": row["method"],
                      "method_id": row["method_id"], "display_method": display[row["method"]],
                      "evidence_stage": "frozen_verified_draft_comparison",
                      "n_train": cfg["n_train"], "dims": "x".join(map(str, cfg["dims"])),
                      "snr": cfg["snr"], "rho": cfg["rho"][0], "n_unique_inputs": size,
                      "pmse_mean": mean, "pmse_sample_sd": sd,
                      "pmse_ci95_low": mean-half, "pmse_ci95_high": mean+half,
                      "exact_successes": successes, "exact_rate": successes/size,
                      "exact_wilson95_low": exact_low, "exact_wilson95_high": exact_high,
                      "ree_mean": float(row["ree_mean"]),
                      "actual_element_f1_mean": float(row["element_f1_mean"]),
                      "rectangular_capacity_mean": float(row["q_rect_mean"])})
        checks.append({"config_id": row["config_id"], "method": row["method"],
                       "unique_denominator": size, "exact_successes": successes,
                       "maximum_summary_difference": float(np.max(np.abs(np.asarray(observed)-saved)))})
    for method in display:
        denominators = sorted(row["n_unique_inputs"] for row in table if row["method"] == method)
        if denominators != [50]*9+[150]:
            raise ValueError("Expected baseline150 and nine configurations50 per method")
    return table, configurations, hashes, checks


def certified_table(summary, unique, configurations):
    """Admit the complete comparison and independently reaggregate plotted fields."""
    methods = {
        'adaptive': ('v52_search_final_original_msic', 'frozen_verified', 'Tensor-ABESS v5.2'),
        'oracle': ('oracle_ols_frozen', 'frozen_verified', 'Oracle OLS'),
        'lasso': ('vector_lasso_cv_isolated_v2', 'certified_new', 'Vector Lasso'),
        'glasso': ('sequential_tensor_glasso_certified_v2', 'certified_new', 'Sequential Tensor-GLasso'),
    }
    expected = {(cid, method) for cid in configurations for method in methods}
    grouped = defaultdict(list)
    keys = set()
    for row in unique:
        key = (row['unique_input_id'], row['method'])
        if key in keys:
            raise ValueError('Duplicate complete-input/method in certified plot data')
        keys.add(key)
        grouped[(row['config_id'], row['method'])].append(row)
    summary_keys = [(row['config_id'], row['method']) for row in summary]
    if len(configurations) != 10 or len(unique) != 2400 or len(summary) != 40 or set(grouped) != expected or set(summary_keys) != expected:
        raise ValueError('All600 inputs,10 configurations and four methods required')
    table, checks = [], []
    for row in summary:
        cid, method = row['config_id'], row['method']
        method_id, stage, label = methods[method]
        members = grouped[(cid, method)]
        if any(item['method_id'] != method_id or item['evidence_stage'] != stage for item in [row, *members]):
            raise ValueError('Uncertified method or wrong evidence stage in plot data')
        values = np.asarray([float(item['pmse']) for item in members])
        exact = np.asarray([float(item['exact_recovery']) for item in members])
        size = len(members)
        if not np.isfinite(values).all() or not np.isin(exact, (0, 1)).all() or any(int(row[key]) != size for key in ('n', 'pmse_valid_n', 'exact_recovery_valid_n')):
            raise ValueError('Invalid values or altered plot denominator')
        successes = int(exact.sum())
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        lo, hi = wilson(successes, size)
        observed = [mean, sd, successes, successes/size, lo, hi]
        saved = [float(row[key]) for key in ('pmse_mean', 'pmse_sd', 'exact_successes', 'exact_recovery_mean', 'exact_wilson_low', 'exact_wilson_high')]
        np.testing.assert_allclose(observed, saved, rtol=1e-11, atol=1e-10)
        cfg = configurations[cid]
        half = float(t.ppf(.975, size-1))*sd/np.sqrt(size)
        table.append({'config_id': cid, 'method': method, 'method_id': method_id,
                      'display_method': label, 'evidence_stage': stage,
                      'n_train': cfg['n_train'], 'dims': 'x'.join(map(str, cfg['dims'])),
                      'snr': cfg['snr'], 'rho': cfg['rho'][0], 'n_unique_inputs': size,
                      'pmse_mean': mean, 'pmse_sample_sd': sd,
                      'pmse_ci95_low': mean-half, 'pmse_ci95_high': mean+half,
                      'exact_successes': successes, 'exact_rate': successes/size,
                      'exact_wilson95_low': lo, 'exact_wilson95_high': hi,
                      'ree_mean': float(row['ree_mean']), 'actual_element_f1_mean': float(row['element_f1_mean']),
                      'actual_element_recall_mean': float(row['element_recall_mean']),
                      'actual_element_fdr_mean': float(row['element_fdr_mean']),
                      'actual_element_nonzero_mean': float(row['element_nonzero_mean']),
                      'mode_f1_macro_mean': float(row['mode_f1_macro_mean']),
                      'rectangular_capacity_mean': float(row['q_rect_mean'])})
        checks.append({'config_id': cid, 'method': method, 'unique_denominator': size,
                       'exact_successes': successes, 'maximum_summary_difference': float(np.max(np.abs(np.asarray(observed)-saved)))})
    for method in methods:
        rows = [item for item in table if item['method'] == method]
        if sorted(item['n_unique_inputs'] for item in rows) != [50]*9+[150]:
            raise ValueError('Expected baseline150 and nine50 configurations per method')
    for cid in configurations:
        identities = [{r['unique_input_id'] for r in grouped[(cid, method)]} for method in methods]
        if any(identities[0] != identity for identity in identities[1:]):
            raise ValueError('Methods are not paired on identical complete inputs')
    return table, checks


def accepted_csv_hashes(root, names):
    audit_name = 'doc/taskcard18/cluster_review/RETURN_VALIDATION.json'
    audit = json.loads((root/audit_name).read_text(encoding='utf8'))
    seal = hashlib.sha256(json.dumps({key: value for key, value in audit.items() if key != 'record_sha256'},
                                    sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf8')).hexdigest()
    if audit.get('status') != 'PASS' or audit.get('blocking_issues') or audit.get('record_sha256') != seal:
        raise ValueError('A sealed passing returned-results audit is required')
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    if any(audit['certified_csv_sha256'].get(name) != value for name, value in hashes.items()):
        raise ValueError('Certified figure CSV differs from the accepted returned-results audit')
    hashes[audit_name] = hashlib.sha256((root/audit_name).read_bytes()).hexdigest()
    return hashes


def certified_data(root):
    _, configurations, hashes, _ = verified_data(root)
    names = ['doc/taskcard18/results/certified_config_summary.csv',
             'doc/taskcard18/results/certified_unique_metrics.csv',
             'doc/taskcard18/results/certified_paired_pmse.csv']
    hashes.update(accepted_csv_hashes(root, names))
    table, checks = certified_table(read_csv(root/names[0]), read_csv(root/names[1]), configurations)
    return table, configurations, hashes, checks


def curve_specs(configurations):
    default = next(cfg for cfg in configurations.values()
                   if cfg["n_train"] == 200 and cfg["dims"] == [32,20,25]
                   and cfg["snr"] == 2 and cfg["rho"] == [0.3]*3)
    factors = [("sample_size", "n_train", "Sample size", "Training sample size", False),
               ("signal_to_noise", "snr", "Signal-to-noise ratio", "Signal-to-noise ratio", False),
               ("tensor_dimension", "dims", "Tensor dimensions", "Tensor dimensions", True),
               ("correlation", "rho", "Predictor correlation", "AR(1) correlation", False)]
    specs = []
    for name, key, title, xlabel, categorical in factors:
        cfg_ids = [cid for cid, cfg in configurations.items()
                   if all(cfg[k] == value for k, value in default.items() if k != key)]
        cfg_ids.sort(key=lambda cid: configurations[cid][key])
        values = [configurations[cid][key] for cid in cfg_ids]
        x = np.arange(len(values), dtype=float) if categorical else np.array(
            [value[0] if key == "rho" else value for value in values], dtype=float)
        labels = ["×".join(map(str,value)) if categorical else f"{position:g}"
                  for position,value in zip(x,values)]
        specs.append({"name": name, "key": key, "title": title, "xlabel": xlabel,
                      "config_ids": cfg_ids, "x": x, "labels": labels})
    return specs


def render_curve(spec, table, destination, qa):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from audit_panel_alignment import require_matplotlib_panel_alignment

    # 183 x82 mm expressed in inches for the plotting backend and source audit.
    fig, axes = plt.subplots(1, 2, figsize=(7.204724409448819,3.2283464566929134))
    fig.subplots_adjust(left=0.086, right=0.98, top=0.77, bottom=0.25, wspace=0.35)
    complete = {row['method'] for row in table} == {'adaptive', 'oracle', 'lasso', 'glasso'}
    methods = {"adaptive": ("Tensor-ABESS v5.2", "#2A6FBB", "o", "-"),
               "oracle": ("Oracle OLS", "#777777", "s", "--")}
    if complete:
        methods = {"adaptive": methods['adaptive'],
                   "lasso": ("Vector Lasso", "#B17A3A", "^", "--"),
                   "glasso": ("Sequential Tensor-GLasso", "#7C6AA5", "D", "-.") ,
                   "oracle": methods['oracle']}
        fig.subplots_adjust(top=0.70)
    x = spec["x"]
    span = max(float(np.ptp(x)), 1.0)
    for order, (method, (label,color,marker,linestyle)) in enumerate(methods.items()):
        rows = [next(row for row in table if row["config_id"] == cid and row["method"] == method)
                for cid in spec["config_ids"]]
        position = x + (order-(len(methods)-1)/2)*span*0.016
        for ax, center, lower, upper in (
            (axes[0], "pmse_mean", "pmse_ci95_low", "pmse_ci95_high"),
            (axes[1], "exact_rate", "exact_wilson95_low", "exact_wilson95_high"),
        ):
            y = np.array([row[center] for row in rows])
            limits = np.array([[row[lower],row[upper]] for row in rows])
            ax.errorbar(position, y, yerr=np.vstack((y-limits[:,0],limits[:,1]-y)),
                        color=color, marker=marker, linestyle=linestyle, markersize=3.5,
                        linewidth=0.9, elinewidth=0.8, capsize=2.4, capthick=0.8,
                        label=label, zorder=3-order)
    for letter, ax in zip(("a","b"),axes):
        ax.annotate(letter, xy=(0,1), xycoords="axes fraction", xytext=(-24,9),
                    textcoords="offset points", ha="left", va="bottom", weight="bold", fontsize=7)
        ax.set_xticks(x, labels=spec["labels"])
        ax.set_xlim(float(x.min())-0.075*span,float(x.max())+0.075*span)
        ax.set_xlabel(spec["xlabel"], labelpad=5)
        ax.tick_params(length=3, width=0.6, pad=3)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_title("PMSE (mean, 95% t CI)", loc="left", fontsize=7, pad=9)
    axes[0].set_ylabel("Prediction MSE", labelpad=6)
    axes[0].margins(y=0.15)
    axes[1].set_title("Exact recovery (95% Wilson CI)", loc="left", fontsize=7, pad=9)
    axes[1].set_ylabel("Rate", labelpad=6)
    axes[1].set_ylim(-0.065,1.085)
    axes[1].set_yticks([0,0.25,0.5,0.75,1], labels=["0","0.25","0.5","0.75","1"])
    fig.text(0.086,0.955,spec["title"], ha="left",va="top",fontsize=7,weight="bold")
    handles = [Line2D([0],[0],color=color,marker=marker,linestyle=linestyle,linewidth=0.9,
                      markersize=3.5,label=label) for label,color,marker,linestyle in methods.values()]
    fig.legend(handles=handles, loc="upper right",bbox_to_anchor=(0.99,0.91 if complete else 0.99),
               ncol=2, handlelength=2.3,columnspacing=1.8,borderpad=0.4)
    fig.text(0.086,0.105,"Certified comparison; Oracle is a known-support diagnostic" if complete else
             "Frozen methods only; complete revised baseline comparison pending", fontsize=7)
    fig.text(0.086,0.035,"R = 150 at baseline; R = 50 otherwise. Each PMSE uses 500 independent test points.", fontsize=7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=qa/f"{spec['name']}.alignment.json",
        overlay_svg=qa/f"{spec['name']}.alignment.svg", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    stem = destination/spec["name"]
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def render_tables(root, table):
    destination = root/"paper/tables"
    destination.mkdir(parents=True, exist_ok=True)
    ordered = sorted(table, key=lambda r:(r["n_train"],r["dims"],r["snr"],r["rho"],r["method"]))
    write_csv(destination/"verified_frozen_results.csv", ordered)
    lines = ["# 已核验冻结方法的配置结果（章节草稿）", "",
             "仅含正式Tensor-ABESS v5.2与Oracle OLS；新增可靠基线完整比较尚未验收。",
             "基准配置为n=200、32×20×25、SNR=2、rho=0.3；合并150套唯一输入，其余各50套。", "",
             "| n / 维度 / SNR / rho | R | ABESS PMSE [95% t CI] | Oracle PMSE [95% t CI] | ABESS Exact [Wilson95%] | Oracle Exact | ABESS REE | ABESS实际元素F1 |",
             "|---|---:|---|---|---|---:|---:|---:|"]
    for adaptive in [row for row in ordered if row["method"] == "adaptive"]:
        oracle = next(row for row in table if row["config_id"] == adaptive["config_id"] and row["method"] == "oracle")
        pmse = lambda row:f"{row['pmse_mean']:.4f} [{row['pmse_ci95_low']:.4f}, {row['pmse_ci95_high']:.4f}]"
        condition = f"{adaptive['n_train']} / {adaptive['dims']} / {adaptive['snr']:g} / {adaptive['rho']:g}"
        exact = f"{adaptive['exact_successes']}/{adaptive['n_unique_inputs']} [{adaptive['exact_wilson95_low']:.3f}, {adaptive['exact_wilson95_high']:.3f}]"
        lines.append(f"| {condition} | {adaptive['n_unique_inputs']} | {pmse(adaptive)} | {pmse(oracle)} | {exact} | {oracle['exact_successes']}/{oracle['n_unique_inputs']} | {adaptive['ree_mean']:.4f} | {adaptive['actual_element_f1_mean']:.4f} |")
    lines += ["", "PMSE含观测噪声；区间反映训练/测试输入重复间的模拟均值不确定性，不把500测试点当500个算法重复。样本SD使用ddof=1，t区间自由度R−1。",
              "Exact为三个声明模式支撑集合同时正确；Oracle直接给真支撑，其Exact=1为定义，Wilson区间见全精度CSV，不解释为可部署方法恢复能力。REE为范数比，未平方；实际元素F1按系数绝对值大于1e−8的集合计算。",
              "各配置分别估计，未进行跨600输入的IID池化推断、p值或多重检验；四图共用同一基准估计。PMSE/Exact已从原冻结逐任务记录独立去重重算并核对；REE/元素指标沿用已核验系数汇总。"]
    (destination/"verified_frozen_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def render_certified_table(root, table):
    destination = root/'paper/tables'
    destination.mkdir(parents=True, exist_ok=True)
    order = {'adaptive': 0, 'lasso': 1, 'glasso': 2, 'oracle': 3}
    ordered = sorted(table, key=lambda row: (row['n_train'], row['dims'], row['snr'], row['rho'], order[row['method']]))
    write_csv(destination/'certified_main_results.csv', ordered)
    lines = ['# 正式四方法配置结果', '',
             '全部600套唯一完整输入、10种配置与四方法均保留；基准配置合并150套，其余各50套。',
             'PMSE列为均值±样本SD，Exact列为成功数/R及Wilson95%区间；全精度CSV另存PMSE均值t区间。', '',
             '| n / 维度 / SNR / rho | 方法 | R | PMSE ± SD | REE | Exact [Wilson95%] | 实际元素F1 | 模式F1 | 实际非零数 | 矩形容量 |',
             '|---|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for row in ordered:
        condition = f"{row['n_train']} / {row['dims']} / {row['snr']:g} / {row['rho']:g}"
        exact = f"{row['exact_successes']}/{row['n_unique_inputs']} [{row['exact_wilson95_low']:.3f}, {row['exact_wilson95_high']:.3f}]"
        lines.append(f"| {condition} | {row['display_method']} | {row['n_unique_inputs']} | {row['pmse_mean']:.4f} ± {row['pmse_sample_sd']:.4f} | {row['ree_mean']:.4f} | {exact} | {row['actual_element_f1_mean']:.4f} | {row['mode_f1_macro_mean']:.4f} | {row['actual_element_nonzero_mean']:.2f} | {row['rectangular_capacity_mean']:.2f} |")
    lines += ['', 'PMSE评价含噪独立响应，每次500个测试点；重复单位是完整训练/测试输入。SD使用ddof=1，均值t区间自由度R−1；各配置单独估计，未池化600输入进行IID推断。',
              '实际元素按|B|>1e−8计算；模式指标使用声明轴支撑，Lasso使用实际非零投影；矩形容量不等于实际非零数。Oracle给定真支撑，Exact=1来自定义，不解释为可部署恢复能力或普遍风险下界。',
              '同输入ABESS减各对照的PMSE差、MCSE及配对t区间见配对结果表；均值图的边际区间重叠不代替配对比较。旧两基线未认证结果保留在历史附录。']
    (destination/'certified_main_results.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


def render_captions(destination):
    text = '''# 四方法主图图注

四图比较正式Tensor-ABESS v5.2（原MSIC；蓝色圆点实线）、折间状态隔离的Vector Lasso（棕色三角虚线）、可靠求解的顺序Tensor-GLasso（紫色菱形点划线）及给定真实支撑的Oracle OLS（灰色方点虚线）。两个更新对照均使用训练内5折CV、原20点lambda网格及全部必要凸拟合的KKT/原始—对偶间隙双证书。顺序Tensor-GLasso的局部CV仍使用完整训练流程确定的其他轴支撑，不属于全流程严格嵌套CV。Oracle不可部署，Exact=1来自给定支撑，不解释为恢复能力或任意估计器的有限样本风险下界。

所有点均保留相应配置的全部唯一完整输入。基准为n=200、32×20×25、SNR=2、三轴rho=0.3，合并150套唯一输入；其余各50套。同一基准在四条曲线复用，未视为额外独立重复。**a** 为含噪独立响应的预测均方误差（PMSE）均值及95% t均值区间，样本SD采用ddof=1，自由度R−1；每次使用500个独立测试点。**b** 为三个声明模式支撑集合同时正确的Exact率及双侧Wilson95%区间。R为完整输入重复数，500测试点不充当500次算法重复；Lasso的声明轴支撑为实际非零元素的轴投影。方法点作轻微水平错开以区分误差棒。各图PMSE纵轴独立设置，Exact纵轴统一；不以边际区间重叠代替同输入配对差判断，不显示p值或多重检验胜出计数。

**图1 | 样本量对预测与支撑恢复的影响。** **a** n=100、200、400下的PMSE；**b** 相应Exact。R依次为50、150、50。n=100时正式算法既有caps(4,4,4)排除真形状(4,3,5)，零完全恢复含有确定的规模覆盖限制；n=400的预测风险与恢复程度分别判读，不能由一个指标代替另一个。

**图2 | 信噪比对预测与支撑恢复的影响。** **a** SNR=0.5、1、2、4下的PMSE；**b** 相应Exact。R依次为50、50、150、50，训练样本量200、维度32×20×25、rho=0.3。SNR变化对应信号方差变化，跨SNR绝对PMSE不解释为相同预测难度下的相对效率；弱信号结果完整保留。

**图3 | 张量维度对预测与支撑恢复的影响。** **a** 16×12×15、32×20×25和64×30×40下的PMSE；**b** 相应Exact。R依次为50、150、50，n=200、SNR=2、rho=0.3，真形状始终(4,3,5)。横轴为有序类别，连线只用于阅读，不代表连续维度标度定律；预测差异与真位置恢复分别评价。

**图4 | 相关性对预测与支撑恢复的影响。** **a** 三轴AR(1)相关系数均为0、0.3、0.6下的PMSE；**b** 相应Exact。R依次为50、150、50，n=200、维度32×20×25、SNR=2。相关坐标替代可能使预测与真实位置恢复呈不同变化，两面板共同限定解释；所有相关性设置均保留。

完整图源见 `certified_figure_source_data.csv`；主表见 `../tables/certified_main_results.csv` 与对应Markdown。同输入配对差、MCSE及t区间另见配对统计表。PDF为183×82mm可编辑矢量，SVG保留文本，PNG为600dpi预览；未作图像增强或AI图像生成。所有区间仅描述固定参数配置下的模拟不确定性，未将600或700标签池化为IID总体。
'''
    (destination/'CAPTIONS.md').write_text(text, encoding='utf8')


def run_qa(skill, source, destination, qa, specs):
    commands = [("source_preflight", ["validate_figure.py",str(source),"--json"])]
    for spec in specs:
        name = spec["name"]
        pdf = str(destination/f"{name}.pdf")
        commands += [(name+".glyph", ["audit_pdf_text.py",pdf,"--min-pt","5","--json"]),
                     (name+".collision", ["audit_figure_collisions.py",pdf,
                                           "--json-out",str(qa/f"{name}.collision.json"),
                                           "--overlay-pdf",str(qa/f"{name}.collision.pdf")])]
    results = []
    environment = dict(os.environ, PYTHONIOENCODING="utf-8")
    for name, arguments in commands:
        result = subprocess.run([sys.executable,"-B",str(skill/"scripts"/arguments[0]),*arguments[1:]],
                                capture_output=True,text=True,encoding="utf-8",env=environment)
        (qa/f"{name}.txt").write_text(result.stdout+result.stderr,encoding="utf-8")
        results.append({"audit":name,"returncode":result.returncode})
        if result.returncode:
            raise RuntimeError(f"Figure QA blocked: {name}; inspect saved audit")
    return results


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--skill-root",type=Path,default=Path.home()/".codex/skills/nature-figure")
    parser.add_argument('--frozen', action='store_true', help='Reproduce the historical two-method draft only')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    qa = root/"doc/taskcard18/figure_qa" / ('frozen_replay' if args.frozen else 'certified')
    qa.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str(qa/"mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    mpl.rcParams.update({"font.family":"sans-serif", "font.sans-serif":["Arial","DejaVu Sans"],
                         "font.size":7,"axes.labelsize":7,"xtick.labelsize":7,"ytick.labelsize":7,
                         "legend.fontsize":7,"pdf.fonttype":42,"svg.fonttype":"none",
                         "axes.spines.right":False,"axes.spines.top":False,"axes.linewidth":0.65,
                         "legend.frameon":False,"figure.facecolor":"white","axes.facecolor":"white"})
    sys.path.insert(0,str(args.skill_root/"scripts"))
    table, configurations, sources, checks = verified_data(root) if args.frozen else certified_data(root)
    specs = curve_specs(configurations)
    destination = root/"paper/figures/frozen_replay" if args.frozen else root/'paper/figures'
    destination.mkdir(parents=True,exist_ok=True)
    if not args.frozen:
        import shutil
        frozen = destination/'frozen'
        if not frozen.exists():
            frozen.mkdir()
            originals = ['CAPTIONS.md', 'frozen_figure_source_data.csv']
            originals += [f"{spec['name']}.{suffix}" for spec in specs for suffix in ('pdf', 'png', 'svg')]
            for name in originals:
                path = destination/name
                if path.exists(): shutil.copy2(path, frozen/name)
            old_qa = root/'doc/taskcard18/figure_qa/FIGURE_VALIDATION.json'
            if old_qa.exists(): shutil.copy2(old_qa, root/'doc/taskcard18/figure_qa/frozen_FIGURE_VALIDATION.json')
    render_tables(root,table) if args.frozen else render_certified_table(root, table)
    source_rows = []
    for spec in specs:
        render_curve(spec,table,destination,qa)
        source_rows += [{"figure":spec["name"],**row} for row in table if row["config_id"] in spec["config_ids"]]
    write_csv(destination/('frozen_figure_source_data.csv' if args.frozen else 'certified_figure_source_data.csv'),source_rows)
    if not args.frozen: render_captions(destination)
    audits = run_qa(args.skill_root,Path(__file__),destination,qa,specs)
    for name,digest in sources.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Read-only figure input changed during rendering")
    report = {"backend":"python","status":"AUTOMATED_QA_PASS_PENDING_VISIBLE_REVIEW",
              "source_sha256":sources,"plot_source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "methods":["adaptive","oracle"] if args.frozen else ['adaptive','lasso','glasso','oracle'],"excluded_by_authorized_scope":["legacy_lasso","legacy_glasso"],
              "selected_method_config_cells":len(table),"unique_inputs_per_method":600,
              "denominators_per_method":[50]*9+[150],"independent_raw_aggregation_checks":checks,
              "dimensions_mm":[183,82],"font_pt":7,"png_dpi":600,"audits":audits,
              "not_complete_revised_baseline_comparison":args.frozen}
    output_files = [destination/f"{spec['name']}.{extension}" for spec in specs for extension in ('pdf','png','svg')]
    if not args.frozen:
        output_files += [destination/'CAPTIONS.md', destination/'certified_figure_source_data.csv',
                         root/'paper/tables/certified_main_results.csv', root/'paper/tables/certified_main_results.md']
    report['output_sha256'] = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in output_files}
    (root/'doc/taskcard18/figure_qa/FIGURE_VALIDATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"FIGURES_RENDERED: four two-panel figures;{len(table)} table rows,600 unique inputs per method; automated QA passed.")


if __name__ == "__main__":
    main()
