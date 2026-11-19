"""Summarise the ProbTalk3D user survey (Google Forms CSV) into charts + JSON."""
import csv, json, sys, statistics as st
from collections import Counter
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
src = sys.argv[1] if len(sys.argv) > 1 else "responses.csv"
KEY = {"B1": "surprised", "B2": "sad", "B3": "happy", "B4": "disgusted", "B5": "angry", "B6": "fear"}
EMO = ["neutral", "happy", "sad", "surprised", "fear", "disgusted", "angry", "contempt"]
rows = list(csv.reader(open(src, encoding="utf-8")))[1:]
rows = [r for r in rows if any(c.strip() for c in r[1:])]
n = len(rows)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
V, P, T = "#7c6cff", "#e45cf0", "#36d6c3"
out = {"respondents": n}
# A: lip-sync + naturalness
A = {}
for i, c in enumerate(["A1", "A2", "A3", "A4"]):
    lip = [int(r[2 + 2 * i]) for r in rows]; nat = [int(r[3 + 2 * i]) for r in rows]
    A[c] = {"lip_sync": round(st.mean(lip), 2), "naturalness": round(st.mean(nat), 2),
            "lip_sd": round(st.stdev(lip), 2), "nat_sd": round(st.stdev(nat), 2)}
allL = [int(r[2 + 2 * i]) for r in rows for i in range(4)]; allN = [int(r[3 + 2 * i]) for r in rows for i in range(4)]
out["A"] = A; out["A_overall"] = {"lip_sync": round(st.mean(allL), 2), "naturalness": round(st.mean(allN), 2)}
# B: emotion recognition
B = {}; conf = np.zeros((6, len(EMO)), int); correct = 0
for k in range(6):
    clip = f"B{k+1}"; ans = [r[10 + 2 * k].strip().lower() for r in rows]
    clar = [int(r[11 + 2 * k]) for r in rows]
    hit = sum(a == KEY[clip] for a in ans); correct += hit
    for a in ans:
        if a in EMO: conf[k, EMO.index(a)] += 1
    B[clip] = {"true": KEY[clip], "accuracy": round(hit / n, 3), "clarity": round(st.mean(clar), 2),
               "most_chosen": Counter(ans).most_common(1)[0][0]}
out["B"] = B; out["B_overall_accuracy"] = round(correct / (6 * n), 3); out["chance"] = round(1 / 8, 3)
# C: diversity perception
C = {c: dict(Counter(r[col].strip() for r in rows)) for c, col in (("C1", 22), ("C2", 24))}
out["C"] = C
out["C_natural"] = {"C1": round(st.mean(int(r[23]) for r in rows), 2), "C2": round(st.mean(int(r[25]) for r in rows), 2)}
json.dump(out, open("survey_summary.json", "w"), indent=2)

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1, 1.25, 1]})
# panel 1
ax = axes[0]; x = np.arange(4); w = .36
l = [A[c]["lip_sync"] for c in A]; na = [A[c]["naturalness"] for c in A]
b1 = ax.bar(x - w / 2, l, w, color=V, label="Lip-sync"); b2 = ax.bar(x + w / 2, na, w, color=P, label="Naturalness")
for b in list(b1) + list(b2): ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .06, f"{b.get_height():.1f}", ha="center", fontsize=10)
ax.set_xticks(x, ["A1\nhappy", "A2\nangry→surpr.", "A3\nsad→happy", "A4\nfear→surpr."], fontsize=9.5)
ax.set_ylim(0, 5.9); ax.set_yticks(range(6)); ax.set_ylabel("Mean rating (1–5)")
ax.set_title(f"Lip-sync {out['A_overall']['lip_sync']:.2f} · naturalness {out['A_overall']['naturalness']:.2f}", fontsize=12)
ax.legend(frameon=False, loc="upper center", ncol=2)
# panel 2
ax = axes[1]
acc = [B[f"B{k+1}"]["accuracy"] * 100 for k in range(6)]
labels = [f"B{k+1}\n{KEY[f'B{k+1}']}" for k in range(6)]
bars = ax.bar(labels, acc, color=[T if a >= 50 else V for a in acc])
for b, a in zip(bars, acc): ax.text(b.get_x() + b.get_width() / 2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=10)
ax.axhline(12.5, ls="--", color="#888"); ax.text(5.45, 14.5, "chance 12.5%", ha="right", color="#666", fontsize=9)
ax.set_ylim(0, 105); ax.set_ylabel("% who picked the right emotion")
ax.set_title(f"Emotion from the face alone: {out['B_overall_accuracy']*100:.0f}% correct (silent clips)", fontsize=12)
# panel 3
ax = axes[2]
cats = ["Exactly the same", "Slightly different", "Clearly different"]; cols = ["#c9c9d6", P, V]
left = np.zeros(2)
for cat, col in zip(cats, cols):
    vals = np.array([C["C1"].get(cat, 0), C["C2"].get(cat, 0)])
    ax.barh(["C1  sampling ON\n(two different takes)", "C2  sampling OFF\n(identical takes)"], vals, left=left, color=col, label=cat)
    for j, v in enumerate(vals):
        if v: ax.text(left[j] + v / 2, j, str(v), ha="center", va="center", color="white" if col != "#c9c9d6" else "#333", fontsize=10)
    left += vals
ax.invert_yaxis(); ax.set_xlabel("Respondents"); ax.set_xlim(0, n)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3, fontsize=9)
ax.set_title("Can people see take-to-take variation?", fontsize=12)
fig.suptitle(f"ProbTalk3D user survey (n = {n})", fontsize=14, fontweight="bold")
plt.tight_layout(); plt.savefig("survey_results.png", dpi=170, bbox_inches="tight")

# confusion matrix
fig, ax = plt.subplots(figsize=(8.5, 4.6))
pct = conf / n * 100
ax.imshow(pct, cmap="Purples", vmin=0, vmax=100, aspect="auto")
for i in range(6):
    for j in range(len(EMO)):
        if conf[i, j]:
            ax.text(j, i, f"{pct[i,j]:.0f}%", ha="center", va="center", color="white" if pct[i, j] > 55 else "#222",
                    fontweight="bold" if EMO[j] == KEY[f"B{i+1}"] else None)
    ax.add_patch(plt.Rectangle((EMO.index(KEY[f"B{i+1}"]) - .5, i - .5), 1, 1, fill=False, ec=T, lw=2.5))
ax.set_xticks(range(len(EMO)), EMO, rotation=25); ax.set_yticks(range(6), [f"B{k+1}: {KEY[f'B{k+1}']}" for k in range(6)])
ax.set_xlabel("What people answered"); ax.set_ylabel("Emotion the model generated")
ax.set_title(f"Emotion confusion matrix (n = {n}); teal box = correct answer", fontsize=12)
for s in ax.spines.values(): s.set_visible(False)
plt.tight_layout(); plt.savefig("survey_confusion.png", dpi=170, bbox_inches="tight")
print(json.dumps(out, indent=1))
