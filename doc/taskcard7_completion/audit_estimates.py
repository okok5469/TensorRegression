"""Independent saved-array metrics and mathematical score audit; no data regeneration."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def array_digest(array):
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def support_metrics(estimated, truth):
    pieces = []
    for found, actual in zip(estimated, truth):
        overlap = len(set(found) & set(actual))
        precision, recall = overlap / len(found), overlap / len(actual)
        f1 = 2 * overlap / (len(found) + len(actual))
        pieces.append((precision, recall, f1))
    result = {f"mode_{key}_{i}": pieces[i-1][j] for i in (1, 2, 3)
              for j, key in enumerate(("precision", "tpr", "f1"))}
    result.update({f"mode_{key}_macro": sum(item[j] for item in pieces) / 3
                   for j, key in enumerate(("precision", "tpr", "f1"))})
    overlap = math.prod(len(set(a) & set(b)) for a, b in zip(estimated, truth))
    q, qtrue = math.prod(map(len, estimated)), math.prod(map(len, truth))
    result.update(element_precision=overlap/q, element_recall=overlap/qtrue,
                  element_f1=2*overlap/(q+qtrue), element_fdr=1-overlap/q,
                  exact_recovery=float(all(set(a) == set(b) for a, b in zip(estimated, truth))),
                  q_hat=q)
    result.update({f"s{i}_hat":len(s) for i,s in enumerate(estimated,1)})
    return result


def audit():
    root = Path(__file__).resolve().parents[2]
    out = root / "numerical_simulation/outputs_v51_attribution/cluster_run"
    plan = read(out / "plan.json")
    errors = dict(support_metrics=0., ree=0., sic=0., loss_identity=0.)
    rows, paired = [], []
    for task in plan["tasks"]:
        name = f"task_{task['task_id']:06d}"
        archive = root / "numerical_simulation/outputs_known_s_full_50/tasks" / (name + ".npz")
        with np.load(archive, allow_pickle=False) as saved:
            truth = saved["B_true"]
            true_supports = [saved[f"S{i}_true"].tolist() for i in (1,2,3)]
            assert array_digest(truth) == task["expected_input_fingerprints"]["B_true"]
            assert array_digest(saved["train_noise"]) == task["expected_input_fingerprints"]["train_noise"]
        selected_models = {}
        for version, folder in (("v5", "outputs_known_s_full_50"),
                                ("v51", "outputs_v51_attribution/cluster_run"),
                                ("v52", "outputs_v52_search_final_full")):
            base = root / "numerical_simulation" / folder / "tasks" / name
            record = read(base.with_suffix(".json"))
            selected = record if version == "v51" else next(x for x in record["results"] if x["method"] == "adaptive")
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as saved:
                suffix = "" if version == "v51" else "_adaptive"
                coefficients = saved["B" + suffix]
                supports = [saved[f"S{i}{suffix}"].tolist() for i in (1,2,3)]
                if version == "v52":
                    assert np.array_equal(saved["B_true"], truth)
                    assert [saved[f"S{i}_true"].tolist() for i in (1,2,3)] == true_supports
                    assert array_digest(saved["train_noise"]) == task["expected_input_fingerprints"]["train_noise"]
            sizes = list(map(len, supports))
            q = math.prod(sizes)
            n = task["config"]["n_train"]
            budget = max(1, math.floor(.7*n))
            assert q <= budget and all(s <= cap for s, cap in zip(sizes, selected["trace"]["s_max"]))
            assert all(s == sorted(set(s)) and min(s) >= 0 and max(s) < dim
                       for s,dim in zip(supports, coefficients.shape))
            assert coefficients.shape == tuple(task["config"]["dims"]) and np.isfinite(coefficients).all()
            mask = np.ones(coefficients.shape, dtype=bool)
            mask[np.ix_(*supports)] = False
            assert np.all(coefficients[mask] == 0)
            metrics = selected["metrics"]
            independently = support_metrics(supports, true_supports)
            for key, value in independently.items():
                error = abs(value-metrics[key]);errors["support_metrics"] = max(errors["support_metrics"],error)
                assert error <= 1e-12, (name,version,key,error)
            ree = np.linalg.norm(coefficients-truth) / max(np.linalg.norm(truth),np.finfo(float).tiny)
            error = abs(ree-metrics["ree"]);errors["ree"] = max(errors["ree"],error)
            assert error <= 1e-12, (name,version,"ree",error)
            rss = metrics["rss_train"]
            loss_error = abs(rss - 2*n*metrics["loss_train"])
            errors["loss_identity"] = max(errors["loss_identity"], loss_error)
            assert loss_error <= 1e-9
            alpha, gamma = task["config"]["msic_refine_alpha"],task["config"]["msic_refine_gamma"]
            loglog = math.log(max(math.log(n),math.e))
            fit = n*math.log(max(rss/n,np.finfo(float).tiny))
            df = alpha*q*loglog
            multiplicity = gamma*loglog*sum(math.log(math.comb(dim,s)) for dim,s in zip(coefficients.shape,sizes))
            error = abs(fit+df+multiplicity-selected["sic"])
            errors["sic"] = max(errors["sic"],error)
            assert error <= 1e-9, (name,version,"sic",error)
            rows.append(dict(task_id=task["task_id"],scenario_id=task["scenario_id"],version=version,
                             q=q,Q=budget,q_over_Q=q/budget,alpha=alpha,gamma=gamma,
                             sic_fit_term=fit,sic_df_term=df,sic_multiplicity_term=multiplicity))
            selected_models[version] = (coefficients, supports)
        left,right=selected_models["v51"],selected_models["v52"]
        paired.append(dict(task_id=task["task_id"],scenario_id=task["scenario_id"],n_train=task["config"]["n_train"],
                           B_exact=np.array_equal(left[0],right[0]),supports_exact=left[1] == right[1]))
    return dict(status="PASS",saved_models_audited=len(rows),max_absolute_errors=errors,
                v51_v52_exact={str(n): dict(count=sum(x["n_train"]==n for x in paired),
                                         B_exact=sum(x["n_train"]==n and x["B_exact"] for x in paired),
                                         supports_exact=sum(x["n_train"]==n and x["supports_exact"] for x in paired))
                              for n in (100,200,400)},
                models=rows,paired_exactness=paired,
                scope="Independent saved-array audit of recovery and coefficient metrics and MSIC formula. RSS is taken from sealed records; loss identity is checked. No X/y or test streams are regenerated, no PMSE/signalMSE/R2/RSS is independently recomputed, no fitting or merging.")


if __name__ == "__main__":
    result = audit()
    destination = Path(__file__).with_name("estimate_audit.json")
    with destination.open("x",encoding="utf-8") as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
        handle.write("\n")
    print(json.dumps({key:value for key,value in result.items() if key not in ("models","paired_exactness")},indent=2))
