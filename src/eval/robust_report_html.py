import argparse
import base64
import os


def _img_b64(path: str) -> str:
    """Convert image file to base64 encoded string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _md_to_html_table(md_path: str) -> str:
    """Convert markdown table to HTML preformatted block.

    Wraps markdown in a styled pre tag for HTML rendering.
    Escapes HTML entities to prevent rendering issues.
    """
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()
    return (
        "<pre style='white-space:pre-wrap; font-family:ui-monospace,Consolas,monospace'>"
        + md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        + "</pre>"
    )


def _section(size_dir: str, heading: str) -> str:
    """Generate HTML section for a specific image size.

    Creates a section with:
    - Delta heatmap
    - FGSM comparison
    - PGD-5, PGD-10, PGD-20 comparisons
    - Delta table (if available)

    Args:
        size_dir: Directory containing the metrics for this size
        heading: Section heading (e.g., "64×64" or "224×224")

    Returns:
        HTML string for the section
    """
    # Define all image paths
    hmap = os.path.join(size_dir, "robust_compare_delta_heatmap.png")
    fgsm = os.path.join(size_dir, "robust_compare_fgsm.png")
    p5 = os.path.join(size_dir, "robust_compare_pgd5.png")
    p10 = os.path.join(size_dir, "robust_compare_pgd10.png")
    p20 = os.path.join(size_dir, "robust_compare_pgd20.png")
    table = os.path.join(size_dir, "robust_compare_delta_table.md")

    # Map of display titles to file paths
    imgs = [
        ("Δ heatmap", hmap),
        ("FGSM", fgsm),
        ("PGD-5", p5),
        ("PGD-10", p10),
        ("PGD-20", p20),
    ]

    # Build HTML blocks
    blocks = [f"<h2>{heading}</h2>"]

    for title, path in imgs:
        if os.path.exists(path):
            blocks.append(f"<h3>{title}</h3>")
            blocks.append(
                f"<img src='data:image/png;base64,{_img_b64(path)}' "
                f"style='max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px'/>"
            )

    # Add table if it exists
    if os.path.exists(table):
        blocks.append("<h3>Δ table (tri − base)</h3>")
        blocks.append(_md_to_html_table(table))

    return "\n".join(blocks)


def main():
    """Main function to generate the robustness comparison HTML report."""
    # Parse command line arguments
    ap = argparse.ArgumentParser(description="Generate HTML report comparing robustness metrics")
    ap.add_argument(
        "--out", default="results/metrics/robust_report.html", help="Output HTML file path"
    )
    ap.add_argument(
        "--title", default="Robustness Comparison — Baseline vs Tri-Objective", help="Report title"
    )
    ap.add_argument(
        "--dir64", default="results/metrics/64", help="Directory containing 64×64 metrics"
    )
    ap.add_argument(
        "--dir224", default="results/metrics/224", help="Directory containing 224×224 metrics"
    )
    args = ap.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Generate HTML head with metadata and styles
    head = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>{args.title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>
 body{{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;
      margin:24px; line-height:1.5}}
 h1,h2,h3{{margin:0.6em 0 0.3em}}
 .note{{background:#f6f8fa; padding:12px 14px;
       border-radius:8px; border:1px solid #e5e7eb}}
</style></head><body>
<h1>{args.title}</h1>
<div class="note"><b>Repro:</b> 64×64 and 224×224 sweeps with <code>--deterministic</code>,
 α=2/255 for PGD.</div>
"""

    # Generate body sections for each available size
    body = []
    if os.path.isdir(args.dir64):
        body.append(_section(args.dir64, "64×64"))
    if os.path.isdir(args.dir224):
        body.append(_section(args.dir224, "224×224"))

    # Combine all parts and write to file
    html = head + "\n".join(body) + "\n</body></html>"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[report] wrote {args.out}")


if __name__ == "__main__":
    main()
