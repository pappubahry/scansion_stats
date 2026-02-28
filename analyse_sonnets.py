# Claude

import os
import re
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt

OMIT_SONNETS = {99, 126, 145}

# --- Parse sonnets from concatenated file ---

all_sonnets_stresses = {}  # sonnet_num -> list of lists of ints (1=stressed, 0=unstressed)

scansion_path = os.path.join(os.path.dirname(__file__), "sonnets_scansion.txt")
with open(scansion_path, "r", encoding="utf-8") as f:
    raw_lines = [line.rstrip() for line in f.readlines()]

sonnet_num = None
stress_lines = []
for line in raw_lines:
    m = re.match(r"^SONNET (\d+)$", line)
    if m:
        if sonnet_num is not None:
            all_sonnets_stresses[sonnet_num] = stress_lines
        sonnet_num = int(m.group(1))
        stress_lines = []
        continue
    if '●' in line or '○' in line:
        stresses = []
        for ch in line:
            if ch == '●':
                stresses.append(1)
            elif ch == '○':
                stresses.append(0)
        stress_lines.append(stresses)
if sonnet_num is not None:
    all_sonnets_stresses[sonnet_num] = stress_lines

# --- Statistics ---

NUM_LINES = 14
NUM_POSITIONS = 10

def compute_stress_stats(sonnets_stresses):
    stress_totals = [[0] * NUM_POSITIONS for _ in range(NUM_LINES)]
    line_counts = [0] * NUM_LINES
    eleventh_syllable_counts = [0] * NUM_LINES

    for sonnet_num in sorted(sonnets_stresses.keys()):
        if sonnet_num in OMIT_SONNETS:
            continue
        stresses = sonnets_stresses[sonnet_num]
        for line_idx, stress in enumerate(stresses):
            if line_idx >= NUM_LINES:
                continue
            line_counts[line_idx] += 1
            for pos in range(min(NUM_POSITIONS, len(stress))):
                stress_totals[line_idx][pos] += stress[pos]
            if len(stress) >= 11:
                eleventh_syllable_counts[line_idx] += 1

    num_sonnets = max(line_counts) if line_counts else 1
    pct_grid = np.zeros((NUM_LINES, NUM_POSITIONS))
    eleventh_pct = np.zeros(NUM_LINES)
    for line_idx in range(NUM_LINES):
        n = line_counts[line_idx]
        if n == 0:
            continue
        for pos in range(NUM_POSITIONS):
            pct_grid[line_idx, pos] = 100.0 * stress_totals[line_idx][pos] / n
        eleventh_pct[line_idx] = 100.0 * eleventh_syllable_counts[line_idx] / num_sonnets

    return pct_grid, eleventh_pct, line_counts

# --- Heatmap ---

def save_heatmap(filename, title, pct_grid, eleventh_pct):
    full_grid = np.column_stack([pct_grid, eleventh_pct])  # 14 x 11
    col_labels = [f"{p+1}" for p in range(NUM_POSITIONS)] + ["11*"]

    fig, ax = plt.subplots(figsize=(8, 7))

    blue_ramp = np.array([
        [1.0,        1.0,        1.0       ],
        [0.88056664, 0.90622658, 0.95643855],
        [0.76371251, 0.81351188, 0.91153345],
        [0.64958777, 0.7217942,  0.86525766],
        [0.53835386, 0.63095344, 0.81758035],
        [0.43017548, 0.54076257, 0.76846675],
        [0.32519538, 0.45078381, 0.71787852],
        [0.22345389, 0.36011489, 0.66577472],
        [0.12458405, 0.2666301,  0.61211437],
        [0.0271575,  0.16368016, 0.55686203],
        [0.0,        0.0,        0.5       ],
    ])
    def cell_color(pct):
        t = pct / 100.0
        if t <= 0.35:
            a = t / 0.35
            return (1.0, 1.0 - a, 1.0 - a)
        elif t <= 0.50:
            blend = (t - 0.35) / 0.15
            red_color = (1.0, 0.0, 0.0)
            return tuple(red_color[i] * (1 - blend) + blue_ramp[0, i] * blend for i in range(3))
        else:
            a = (t - 0.65) / 0.35
            idx = a * (len(blue_ramp) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(blue_ramp) - 1)
            frac = idx - lo
            return tuple(blue_ramp[lo, i] * (1 - frac) + blue_ramp[hi, i] * frac for i in range(3))

    def eleventh_color(pct):
        g = 1.0 - min(pct / 40.0, 1.0) * 0.5
        return (g, g, g)

    avg_row = full_grid.mean(axis=0)

    gap = 0.3
    avg_y = -1 - gap

    def draw_cell(col, y, val):
        fc = eleventh_color(val) if col == NUM_POSITIONS else cell_color(val)
        ax.add_patch(plt.Rectangle(
            (col, y), 1, 1,
            facecolor=fc,
            edgecolor="white", linewidth=1.5,
        ))
        text_color = "white" if val > 80 else "black"
        text = "100" if val > 99.9 else f"{val:.1f}"
        ax.text(col + 0.5, y + 0.5, text,
                ha="center", va="center", fontsize=10,
                color=text_color)

    for row in range(NUM_LINES):
        for col in range(full_grid.shape[1]):
            draw_cell(col, NUM_LINES - 1 - row, full_grid[row, col])

    for col in range(full_grid.shape[1]):
        draw_cell(col, avg_y, avg_row[col])

    ax.set_xlim(0, full_grid.shape[1])
    ax.set_ylim(avg_y, NUM_LINES)
    ax.set_xticks([c + 0.5 for c in range(full_grid.shape[1])])
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(col_labels, fontsize=12)
    row_labels = [str(i + 1) for i in range(NUM_LINES)] + ["Average"]
    row_labels[0] = "Line 1"
    yticks = [NUM_LINES - 1 - r + 0.5 for r in range(NUM_LINES)] + [avg_y + 0.5]
    ax.set_yticks(yticks)
    ax.set_yticklabels(row_labels, fontsize=12)
    title_lines = title.split("\n", 1)
    fig.canvas.draw()
    first_tick = ax.get_xticklabels()[0]
    fb = first_tick.get_window_extent().transformed(ax.transData.inverted())
    ax.annotate("Syllable", xy=(fb.x0 - 0.15, (fb.y0 + fb.y1) / 2), xycoords="data",
                fontsize=12, ha="right", va="center",
                annotation_clip=False)
    ax_center = (ax.get_position().x0 + ax.get_position().x1) / 2
    fig.suptitle(title_lines[0], fontsize=22, x=ax_center, y=1.01)
    if len(title_lines) > 1:
        ax.set_title(title_lines[1], fontsize=16, pad=16)
    else:
        ax.set_title("", pad=12)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    ax.tick_params(pad=8, axis="y")
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_x = 1.0
    for text_obj in [fig._suptitle, ax.title] + list(ax.get_yticklabels()):
        if text_obj and text_obj.get_text():
            bb = text_obj.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            left_x = min(left_x, bb.x0)

    bottom_text = "Scansion is according to me; your results may vary.\nN = 151. Sonnets 99, 126, and 145 excluded.\n* Column 11 shows percentages of feminine line endings."

    fig.text(left_x, -0.01, bottom_text, fontsize=12,
             ha='left', va='bottom', transform=fig.transFigure)
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved heatmap to {filename}")

# --- Irregularity scores ---

def compute_irregularity(sonnets_stresses, pct_grid):
    avg = pct_grid / 100.0
    scores = {}
    for sonnet_num in sorted(sonnets_stresses.keys()):
        if sonnet_num in OMIT_SONNETS:
            continue
        stresses = sonnets_stresses[sonnet_num]
        score = 0.0
        for line_idx, stress in enumerate(stresses):
            if line_idx >= NUM_LINES:
                continue
            for pos in range(min(NUM_POSITIONS, len(stress))):
                score += abs(stress[pos] - avg[line_idx, pos])
        scores[sonnet_num] = score
    return scores

def compute_double_unstressed_pct(sonnets_stresses):
    scores = {}
    for sonnet_num in sorted(sonnets_stresses.keys()):
        if sonnet_num in OMIT_SONNETS:
            continue
        total_intervals = 0
        double_intervals = 0
        for line_idx, stress in enumerate(sonnets_stresses[sonnet_num]):
            if line_idx >= NUM_LINES:
                continue
            stressed_positions = [p for p in range(min(NUM_POSITIONS, len(stress))) if stress[p] == 1]
            for i in range(len(stressed_positions) - 1):
                gap = stressed_positions[i + 1] - stressed_positions[i] - 1
                total_intervals += 1
                if gap == 2:
                    double_intervals += 1
        scores[sonnet_num] = 100.0 * double_intervals / total_intervals if total_intervals else 0.0
    return scores

# --- Run ---

# Original stats
orig_pct, orig_11th, _ = compute_stress_stats(all_sonnets_stresses)
save_heatmap("heatmap_original.png", "Iambic pentameter in Shakespeare's sonnets\nPercentage stressed beat (ictic) at each syllable", orig_pct, orig_11th)

total_stresses = sum(sum(s[:10]) for sn, stresses in all_sonnets_stresses.items() if sn not in OMIT_SONNETS for s in stresses)
total_lines = sum(len(stresses) for sn, stresses in all_sonnets_stresses.items() if sn not in OMIT_SONNETS)
print(f"Average stresses per line (first 10 syllables): {total_stresses / total_lines:.2f} ({total_stresses} stresses across {total_lines} lines)")

# Apply stress modifications
modified_stresses = copy.deepcopy(all_sonnets_stresses)
for sonnet_num, stresses in modified_stresses.items():
    for stress in stresses:
        for i in range(1, len(stress) - 1):
            if stress[i - 1] == 0 and stress[i] == 0 and stress[i + 1] == 0:
                stress[i] = 1
        if len(stress) == 10 and stress[-1] == 0 and stress[-2] == 0:
            stress[-1] = 1

MANUAL_OVERRIDES = [
    (4, 1, 6),
    (4, 7, 6),
    (116, 2, 6),
]
for sonnet_num, line_num, syl_num in MANUAL_OVERRIDES:
    modified_stresses[sonnet_num][line_num - 1][syl_num - 1] = 1

# Modified stats
mod_pct, mod_11th, _ = compute_stress_stats(modified_stresses)
save_heatmap("heatmap_modified.png", "Iambic pentameter in Shakespeare's sonnets\nPercentage ictic at each syllable", mod_pct, mod_11th)

mod_total_stresses = sum(sum(s[:10]) for sn, stresses in modified_stresses.items() if sn not in OMIT_SONNETS for s in stresses)
mod_total_lines = sum(len(stresses) for sn, stresses in modified_stresses.items() if sn not in OMIT_SONNETS)
print(f"Average stresses per line (first 10 syllables, after modifications): {mod_total_stresses / mod_total_lines:.2f} ({mod_total_stresses} stresses across {mod_total_lines} lines)")

# Irregularity scores
orig_irreg = compute_irregularity(all_sonnets_stresses, orig_pct)
mod_irreg = compute_irregularity(modified_stresses, mod_pct)
mod_double = compute_double_unstressed_pct(modified_stresses)

csv_path = os.path.join(os.path.dirname(__file__), "irregularity_scores.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sonnet", "irregularity_original", "irregularity_modified", "pct_double_unstressed"])
    for sonnet_num in sorted(orig_irreg.keys()):
        writer.writerow([sonnet_num, f"{orig_irreg[sonnet_num]:.2f}", f"{mod_irreg[sonnet_num]:.2f}",
                         f"{mod_double[sonnet_num]:.1f}"])
print(f"Saved irregularity scores to {csv_path}")
