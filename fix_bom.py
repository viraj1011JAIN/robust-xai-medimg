# fix_bom.py
"""Remove BOM (Byte Order Mark) from robust_report_html.py"""

filename = "src/eval/robust_report_html.py"

# Read as binary
with open(filename, "rb") as f:
    content = f.read()

# Check for and remove BOM
if content.startswith(b"\xef\xbb\xbf"):
    print(f"✓ BOM found in {filename}, removing...")
    content = content[3:]

    # Write back without BOM
    with open(filename, "wb") as f:
        f.write(content)

    print(f"✓ BOM removed from {filename}")
else:
    print(f"✓ No BOM found in {filename}")
