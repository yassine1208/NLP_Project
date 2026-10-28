"""Create slide-ready SVG charts and score tables from the saved 20 Newsgroups run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "presentation_assets"
OUT.mkdir(exist_ok=True)

evaluation = json.loads((ARTIFACTS / "evaluation_results.json").read_text(encoding="utf-8"))
timings = evaluation.get("training_timings_seconds", {})
names = ["LDA_BoW (Controlled & Practical)", "NMF_BoW (Controlled)", "NMF_TFIDF (Practical)"]
short = ["LDA · BoW", "NMF · BoW", "NMF · TF-IDF"]
colors = ["#355C7D", "#2A7F75", "#D17A32"]
rows = []
for model, label, timing_key in zip(names, short, ["lda_bow_seconds", "nmf_bow_seconds", "nmf_tfidf_seconds"]):
    item = evaluation[model]
    align = item.get("external_alignment") or {}
    rows.append({
        "Model": model,
        "Display name": label,
        "C_v coherence": item["coherence_c_v"],
        "U_mass coherence": item["coherence_u_mass"],
        "Topic diversity": item["topic_diversity"],
        "Redundancy rate": item["redundancy_rate"],
        "ARI": align.get("adjusted_rand_index"),
        "NMI": align.get("normalized_mutual_information"),
        "Majority category purity": align.get("majority_category_purity"),
        "Documents scored": align.get("documents_scored"),
        "Training seconds": timings.get(timing_key),
    })

with (OUT / "model_performance.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

with (OUT / "per_topic_coherence.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Topic", "C_v coherence"])
    for model, label in zip(names, short):
        for index, score in enumerate(evaluation[model].get("coherence_c_v_per_topic", []), start=1):
            writer.writerow([label, index, score])


def svg_start(title: str, subtitle: str, width: int = 1200, height: int = 700) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#19232D}.muted{fill:#66727D}.grid{stroke:#E1E6EA;stroke-width:1}</style>',
        f'<text x="64" y="72" font-size="34" font-weight="700">{title}</text>',
        f'<text x="64" y="111" font-size="18" class="muted">{subtitle}</text>',
    ]


def write_bars(filename: str, title: str, subtitle: str, column: str, fmt: str, max_value: float, note: str = "") -> None:
    bits = svg_start(title, subtitle)
    left, top, full_width, row_gap = 300, 190, 760, 115
    for i, (row, label, color) in enumerate(zip(rows, short, colors)):
        y = top + i * row_gap
        value = row[column]
        width = max(0, min(full_width, full_width * value / max_value))
        bits.extend([
            f'<text x="64" y="{y+30}" font-size="23" font-weight="600">{label}</text>',
            f'<rect x="{left}" y="{y}" width="{full_width}" height="42" rx="4" fill="#EEF1F3"/>',
            f'<rect x="{left}" y="{y}" width="{width:.1f}" height="42" rx="4" fill="{color}"/>',
            f'<text x="{left+width+14:.1f}" y="{y+29}" font-size="21" font-weight="600">{fmt.format(value)}</text>',
        ])
    bits.append(f'<text x="64" y="590" font-size="17" class="muted">{note}</text>')
    bits.append('</svg>')
    (OUT / filename).write_text("\n".join(bits), encoding="utf-8")


write_bars("coherence_cv.svg", "Topic coherence (Cᵥ)", "Higher values indicate more coherent top-word co-occurrence.", "C_v coherence", "{:.4f}", 0.75,
           "Best: NMF-BoW (0.6458). The comparison is unsupervised and uses the held-out text for coherence estimation.")
write_bars("topic_diversity.svg", "Topic-word diversity", "Share of unique terms across all top-word lists; higher means less repetition.", "Topic diversity", "{:.1%}", 1.0,
           "Best: NMF-TF-IDF (96.0%); LDA 90.5%; NMF-BoW 89.5%.")
bits = svg_start("U-Mass coherence", "Values closer to zero are better; bar position is mapped from a shared −5 to 0 scale.")
left, top, full_width, row_gap = 300, 190, 760, 115
for i, (row, label, color) in enumerate(zip(rows, short, colors)):
    y = top + i * row_gap
    value = row["U_mass coherence"]
    width = max(0, min(full_width, full_width * (value + 5) / 5))
    bits.extend([
        f'<text x="64" y="{y+30}" font-size="23" font-weight="600">{label}</text>',
        f'<rect x="{left}" y="{y}" width="{full_width}" height="42" rx="4" fill="#EEF1F3"/>',
        f'<rect x="{left}" y="{y}" width="{width:.1f}" height="42" rx="4" fill="{color}"/>',
        f'<text x="{left+width+14:.1f}" y="{y+29}" font-size="21" font-weight="600">{value:.4f}</text>',
    ])
bits.extend(['<text x="64" y="590" font-size="17" class="muted">Best (closest to zero): NMF-BoW (−2.0231). U-Mass differs from Cᵥ in scale and direction.</text>', '</svg>'])
(OUT / "coherence_umass.svg").write_text("\n".join(bits), encoding="utf-8")
write_bars("training_time.svg", "Model training time", "Recorded fit time for this run; hardware dependent, lower is faster.", "Training seconds", "{:.2f} s", 50,
           "NMF-TF-IDF was fastest at 13.73 s. LDA used its full 40-iteration limit.")

# Category-alignment comparison uses grouped bars with a shared 0–1 scale.
bits = svg_start("Alignment with original categories", "External evaluation only: labels were not used to fit or select topics.")
align_keys = [("ARI", "ARI"), ("NMI", "NMI"), ("Majority category purity", "Purity")]
left, top, full_width, row_gap = 300, 175, 700, 140
for ri, (row, label) in enumerate(zip(rows, short)):
    y = top + ri * row_gap
    bits.append(f'<text x="64" y="{y+25}" font-size="22" font-weight="600">{label}</text>')
    for ci, (key, metric_label) in enumerate(align_keys):
        yy = y + ci * 31
        value = row[key]
        width = full_width * value
        bits.extend([
            f'<text x="145" y="{yy+18}" font-size="15" class="muted">{metric_label}</text>',
            f'<rect x="300" y="{yy}" width="{full_width}" height="22" rx="3" fill="#EEF1F3"/>',
            f'<rect x="300" y="{yy}" width="{width:.1f}" height="22" rx="3" fill="{colors[ci]}"/>',
            f'<text x="{310+width:.1f}" y="{yy+17}" font-size="15" font-weight="600">{value:.3f}</text>',
        ])
bits.append('<text x="64" y="620" font-size="17" class="muted">Purity is the weighted share of documents in each topic’s largest category; it is not classification accuracy.</text>')
bits.append('</svg>')
(OUT / "category_alignment.svg").write_text("\n".join(bits), encoding="utf-8")

# Per-topic C_v gives the audience the spread hidden by model averages.
bits = svg_start("Coherence varies by topic", "Cᵥ for each model-internal topic; topic numbers do not identify the same theme across models.")
chart = {"x": 90, "y": 180, "w": 1030, "h": 350}
for tick in range(0, 10):
    value = tick / 10
    yy = chart["y"] + chart["h"] * (1 - value)
    bits.extend([f'<line x1="{chart["x"]}" y1="{yy:.1f}" x2="{chart["x"]+chart["w"]}" y2="{yy:.1f}" class="grid"/>',
                 f'<text x="48" y="{yy+6:.1f}" font-size="14" class="muted">{value:.1f}</text>'])
for mi, (model, label, color) in enumerate(zip(names, short, colors)):
    vals = evaluation[model].get("coherence_c_v_per_topic", [])
    points = []
    for i, score in enumerate(vals):
        x = chart["x"] + chart["w"] * i / max(1, len(vals)-1)
        y = chart["y"] + chart["h"] * (1 - score)
        points.append(f"{x:.1f},{y:.1f}")
        bits.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    bits.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" opacity="0.75"/>')
    lx = 300 + mi * 220
    bits.extend([f'<line x1="{lx}" y1="595" x2="{lx+36}" y2="595" stroke="{color}" stroke-width="4"/>',
                 f'<text x="{lx+46}" y="602" font-size="17">{label}</text>'])
bits.extend(['<text x="90" y="565" font-size="14" class="muted">Topic 1</text>',
             '<text x="1050" y="565" font-size="14" class="muted">Topic 20</text>', '</svg>'])
(OUT / "per_topic_coherence.svg").write_text("\n".join(bits), encoding="utf-8")

print(f"Created {len(list(OUT.glob('*.svg')))} SVG charts and 2 CSV score tables in {OUT}")
