"""Phase C: full Tensor-ABESS, baselines, metrics, reproducible benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensor_abess.baselines import run_all_baselines
from tensor_abess.metrics import evaluate_support
from tensor_abess.tensor import tensor_abess
from tensor_abess.utils.data import make_tensor_data


def run_benchmark(seed: int = 2026) -> dict:
    data = make_tensor_data(
        n=200,
        p1=10,
        p2=5,
        p3=8,
        s1=3,
        s2=2,
        s3=2,
        snr=25.0,
        random_state=seed,
    )
    X, y = data["X"], data["y"]

    # Narrow grid around truth for speed; full grid in phase D
    s1_grid = [2, 3, 4]
    s2_grid = [1, 2, 3]
    s3_grid = [1, 2, 3]

    result = tensor_abess(
        X,
        y,
        s1_grid=s1_grid,
        s2_grid=s2_grid,
        s3_grid=s3_grid,
        init_strategy="marginal",
        random_state=seed,
    )

    metrics = evaluate_support(
        result.S1,
        result.S2,
        result.S3,
        data["S1"],
        data["S2"],
        data["S3"],
        result.B,
        data["B"],
        data["p1"],
        data["p2"],
        data["p3"],
    )

    rows = []
    rows.append(
        {
            "method": "tensor_abess",
            "s_hat": [result.s1, result.s2, result.s3],
            "s_true": [data["s1"], data["s2"], data["s3"]],
            "exact": metrics.exact_rectangular,
            "frob": metrics.frobenius_error,
            "loss": result.loss,
            "sic": result.sic,
        }
    )

    for bl in run_all_baselines(
        X,
        y,
        cp_rank=3,
        tucker_ranks=(3, 2, 2),
        random_state=seed,
    ):
        m = evaluate_support(
            bl.S1,
            bl.S2,
            bl.S3,
            data["S1"],
            data["S2"],
            data["S3"],
            bl.B,
            data["B"],
            data["p1"],
            data["p2"],
            data["p3"],
        )
        rows.append(
            {
                "method": bl.name,
                "s_hat": [bl.S1.size, bl.S2.size, bl.S3.size],
                "s_true": [data["s1"], data["s2"], data["s3"]],
                "exact": m.exact_rectangular,
                "frob": m.frobenius_error,
                "loss": bl.loss,
            }
        )

    return {
        "seed": seed,
        "true_support": {
            "S1": data["S1"].tolist(),
            "S2": data["S2"].tolist(),
            "S3": data["S3"].tolist(),
        },
        "tensor_abess": {
            "S1": result.S1.tolist(),
            "S2": result.S2.tolist(),
            "S3": result.S3.tolist(),
            "loss_history": result.loss_history,
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase C reproducible benchmark")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    out = run_benchmark(seed=args.seed)

    print("=" * 70)
    print(f"Phase C benchmark  (seed={args.seed})")
    print("=" * 70)
    print(f"True support: S1={out['true_support']['S1']}, "
          f"S2={out['true_support']['S2']}, S3={out['true_support']['S3']}")
    print()
    print(f"{'Method':<30} {'s_hat':<12} {'Exact':<8} {'Frob':<10} {'Loss':<10}")
    print("-" * 70)
    for row in out["results"]:
        s_hat = row["s_hat"]
        loss = row.get("loss", float("nan"))
        print(
            f"{row['method']:<30} {str(s_hat):<12} "
            f"{str(row['exact']):<8} {row['frob']:<10.4f} {loss:<10.4f}"
        )

    if args.output:
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
