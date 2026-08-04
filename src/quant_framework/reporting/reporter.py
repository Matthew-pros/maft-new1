"""
12 Reporting – basic reporter (legacy) + compatibility
"""

from pathlib import Path
import json
import base64
from typing import Dict
import pandas as pd

class Reporter:
    """Simple reporter – embeds PNG + metrics into HTML, deterministic"""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, data: Dict, filename: str) -> Path:
        if not isinstance(data, dict):
            raise TypeError("data must be dict")
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        return path

    def embed_png_as_html(self, png_path: str, metrics: Dict, html_path: str, title: str = "Report") -> Path:
        png_path = Path(png_path)
        html_path = self.output_dir / Path(html_path).name
        if png_path.exists():
            b64 = base64.b64encode(png_path.read_bytes()).decode()
            img_tag = f'<img src="data:image/png;base64,{b64}" style="width:100%;max-width:1200px;">'
        else:
            img_tag = "<p>Image not found</p>"
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>\n" for k in sorted(metrics.keys()) if not k.endswith('_raw') for v in [metrics[k]])
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{background:#0b0e14;color:#e6edf3;font-family:sans-serif;padding:20px}} .card{{background:#161b22;border:1px solid #30363d;padding:16px;border-radius:8px;margin:16px 0}} table{{border-collapse:collapse;width:100%}} th{{background:#21262d;color:#58a6ff;padding:8px;text-align:left}} td{{padding:8px;border-bottom:1px solid #21262d}}</style></head><body>
<h1>{title}</h1><div class="card">{img_tag}</div><div class="card"><table><tr><th>Metric</th><th>Value</th></tr>{rows}</table></div></body></html>"""
        html_path.write_text(html, encoding='utf-8')
        return html_path
