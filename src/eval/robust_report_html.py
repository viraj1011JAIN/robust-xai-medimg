import argparse
import base64
import os


def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _md_to_html_table(md_path: str) -> str:
    # very simple: wrap the markdown preformatted; GitHub renders md, but for HTML we keep it readable
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()
    return (
        "<pre style='white-space:pre-wrap; font-family:ui-monospace,Consolas,monospace'>"
        + md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre>"
    )


def _section(size_dir: str, heading: str) -> str:
    hmap = os.path.join(size_dir, "robust_compare_delta_heatmap.png")
    fgsm = os.path.join(size_dir, "robust_compare_fgsm.png")
    p5 = os.path.join(size_dir, "robust_compare_pgd5.png")
    p10 = os.path.join(size_dir, "robust_compare_pgd10.png")
    p20 = os.path.join(size_dir, "robust_compare_pgd20.png")
    table = os.path.join(size_dir, "robust_compare_delta_table.md")

    imgs = [
        ("Î” heatmap", hmap),
        ("FGSM", fgsm),
        ("PGD-5", p5),
        ("PGD-10", p10),
        ("PGD-20", p20),
    ]
    blocks = [f"<h2>{heading}</h2>"]
    for title, path in imgs:
        if os.path.exists(path):
            blocks.append(f"<h3>{title}</h3>")
            blocks.append(
                f"<img src='data:image/png;base64,{_img_b64(path)}' style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px'/>"  # noqa: E501
            )
    if os.path.exists(table):
        blocks.append("<h3>Î” table (tri âˆ’ base)</h3>")
        blocks.append(_md_to_html_table(table))
    return "\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/metrics/robust_report.html")
    ap.add_argument(
        "--title", default="Robustness Comparison â€” Baseline vs Tri-Objective"
    )
    ap.add_argument("--dir64", default="results/metrics/64")
    ap.add_argument("--dir224", default="results/metrics/224")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    head = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>{args.title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body{{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif; margin:24px; line-height:1.5}}  # noqa: E501
 h1,h2,h3{{margin:0.6em 0 0.3em}}
 .note{{background:#f6f8fa; padding:12px 14px; border-radius:8px; border:1px solid #e5e7eb}}
</style></head><body>
<h1>{args.title}</h1>
<div class="note"><b>Repro:</b> 64Ã—64 and 224Ã—224 sweeps with <code>--deterministic</code>, Î±=2/255 for PGD.</div>
"""
    body = []
    if os.path.isdir(args.dir64):
        body.append(_section(args.dir64, "64Ã—64"))
    if os.path.isdir(args.dir224):
        body.append(_section(args.dir224, "224Ã—224"))

    html = head + "\n".join(body) + "\n</body></html>"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
