from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _norm_key(k: str) -> str:
    return k.strip().lower().replace(" ", "_")


def _extract_common_schema(rows: List[Dict[str, str]]):
    """Make column names robust:
    - accept auroc or auc (case-insensitive)
    - accept img_size or img-size (ignored for compare)
    - accept attack names like 'pgd10' or 'pgd' + steps==10
    """
    normed: List[Dict[str, str]] = []
    for r in rows:
        nr = {_norm_key(k): v for k, v in r.items()}
        # standardize auroc
        if "auroc" not in nr:
            if "auc" in nr:
                nr["auroc"] = nr["auc"]
        # standardize eps, steps
        if "eps" in nr:
            try:
                nr["eps"] = float(nr["eps"])
            except Exception:
                pass
        if "steps" in nr:
            try:
                nr["steps"] = int(float(nr["steps"]))
            except Exception:
                pass
        # infer steps from attack like 'pgd10' if missing
        if "attack" in nr and ("steps" not in nr or nr["steps"] in ("", None)):
            a = str(nr["attack"]).lower()
            if a.startswith("pgd"):
                s = "".join([c for c in a[3:] if c.isdigit()])
                if s:
                    try:
                        nr["steps"] = int(s)
                    except Exception:
                        pass
        # ensure auroc float
        if "auroc" in nr:
            try:
                nr["auroc"] = float(nr["auroc"])
            except Exception:
                nr["auroc"] = float("nan")
        normed.append(nr)
    return normed


def _group_key(r: Dict[str, str]) -> Tuple[str, int, float]:
    attack = str(r.get("attack", "")).lower()
    steps = int(r.get("steps", 0) or 0)
    eps = float(r.get("eps", 0.0) or 0.0)
    return (attack, steps, eps)


def _align(base_rows, tri_rows):
    """Align rows on (attack, steps, eps). Return dict key->(base_auroc, tri_auroc)."""
    base_map: Dict[Tuple[str, int, float], float] = {}
    tri_map: Dict[Tuple[str, int, float], float] = {}
    for r in base_rows:
        if "auroc" in r:
            base_map[_group_key(r)] = float(r["auroc"])
    for r in tri_rows:
        if "auroc" in r:
            tri_map[_group_key(r)] = float(r["auroc"])
    keys = sorted(
        set(base_map.keys()) | set(tri_map.keys()), key=lambda t: (t[0], t[1], t[2])
    )
    out: Dict[Tuple[str, int, float], Tuple[float, float]] = {}
    for k in keys:
        out[k] = (base_map.get(k, float("nan")), tri_map.get(k, float("nan")))
    return out


def _write_delta_table_md(aligned, out_md: Path):
    # Write a compact markdown with grouped sections per attack+steps
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Robustness Δ (tri - base)\n\n")
        current = None
        block = []
        for (attack, steps, eps), (b, t) in aligned.items():
            if current != (attack, steps):
                if block:
                    # flush previous block
                    f.write("\n".join(block) + "\n\n")
                    block = []
                f.write(f"## {attack.upper()} (steps={steps})\n\n")
                block.append("| eps | base_auroc | tri_auroc | delta |")
                block.append("|-----:|-----------:|----------:|------:|")
                current = (attack, steps)
            delta = (t - b) if (t == t and b == b) else float("nan")  # NaN-safe
            block.append(f"| {eps:.0f} | {b:.3f} | {t:.3f} | {delta:.3f} |")
        if block:
            f.write("\n".join(block) + "\n")


def _plot_heatmap(aligned, out_png: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return  # plotting optional

    # Build grid for a single representative attack (PGD with max steps per eps)
    pgd_keys = [(k, v) for k, v in aligned.items() if k[0].startswith("pgd")]
    if not pgd_keys:
        return
    # choose the highest steps per eps
    by_eps: Dict[float, Tuple[Tuple[str, int, float], Tuple[float, float]]] = {}
    for (attack, steps, eps), vals in pgd_keys:
        cur = by_eps.get(eps)
        if (cur is None) or (steps > cur[0][1]):
            by_eps[eps] = ((attack, steps, eps), vals)
    eps_list = sorted(by_eps.keys())
    data = []
    for eps in eps_list:
        base, tri = by_eps[eps][1]
        delta = (tri - base) if (tri == tri and base == base) else np.nan
        data.append(delta)
    arr = np.array(data)[None, :]  # 1 x E
    fig = plt.figure(figsize=(max(4, len(eps_list)), 2.2))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest")
    plt.yticks([0], ["Δ AUROC (tri - base)"])
    plt.xticks(range(len(eps_list)), [f"{int(e)}" for e in eps_list])
    plt.xlabel("epsilon (1/255)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_pgd10_lines(aligned, out_png: Path):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    xs, y_base, y_tri = [], [], []
    for (attack, steps, eps), (b, t) in aligned.items():
        if str(attack).lower().startswith("pgd") and int(steps) == 10:
            xs.append(float(eps))
            y_base.append(float(b))
            y_tri.append(float(t))
    if not xs:
        return
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    y_base = [y_base[i] for i in order]
    y_tri = [y_tri[i] for i in order]
    fig = plt.figure(figsize=(5.5, 3.2))
    plt.plot(xs, y_base, marker="o", label="baseline")
    plt.plot(xs, y_tri, marker="o", label="tri-obj")
    plt.xlabel("epsilon (1/255)")
    plt.ylabel("AUROC")
    plt.title("PGD-10 comparison")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Baseline sweep CSV")
    ap.add_argument("--tri", required=True, help="Tri-obj sweep CSV")
    ap.add_argument(
        "--outdir", default="results/metrics", help="Output directory for plots/tables"
    )
    args = ap.parse_args()

    base = Path(args.base)
    tri = Path(args.tri)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_rows = _extract_common_schema(_read_csv_rows(base))
    tri_rows = _extract_common_schema(_read_csv_rows(tri))
    aligned = _align(base_rows, tri_rows)

    # Artifacts
    _plot_heatmap(aligned, outdir / "robust_compare_delta_heatmap.png")
    _plot_pgd10_lines(aligned, outdir / "robust_compare_pgd10.png")
    _write_delta_table_md(aligned, outdir / "robust_compare_delta_table.md")

    print(f"[compare] wrote plots & table in {outdir}")


if __name__ == "__main__":  # pragma: no cover
    main()
