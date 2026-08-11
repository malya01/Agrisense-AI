"""
build_dashboard.py
-------------------
Patches outputs/dashboard.html in place with the latest
outputs/dashboard_data.json, so the HTML stays a single portable file
you can double-click or email -- no local server, no fetch() needed.

Run this AFTER export_dashboard_data.py.
Usage (from inside src/):
    python3 build_dashboard.py
"""

import re
import json

HTML_PATH = "../outputs/dashboard.html"
DATA_PATH = "../outputs/dashboard_data.json"


def build():
    with open(DATA_PATH) as f:
        data_json = f.read()
        json.loads(data_json)  # sanity check it's valid JSON before writing

    with open(HTML_PATH, encoding="utf-8") as f:
        html = f.read()

    pattern = r'(<script id="data-block" type="application/json">)(.*?)(</script>)'
    new_html, n = re.subn(pattern, lambda m: m.group(1) + data_json + m.group(3), html, flags=re.S)

    if n == 0:
        raise RuntimeError("Could not find the data-block <script> tag in dashboard.html")

    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(new_html)

    print(f"Updated {HTML_PATH} with fresh data ({len(data_json)} bytes)")


if __name__ == "__main__":
    build()
