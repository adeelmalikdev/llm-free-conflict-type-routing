import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#444444"})

# ---------- Figure 1: EM and F1 per arm ----------
arms   = ["Oracle router", "Learned router", "Always-multi", "Always-single"]
em     = [0.524, 0.454, 0.428, 0.190]
em_ci  = [(0.480, 0.568), (0.410, 0.496), (0.386, 0.472), (0.156, 0.226)]
f1     = [0.796, 0.767, 0.847, 0.647]
f1_ci  = [(0.774, 0.816), (0.745, 0.788), (0.832, 0.861), (0.629, 0.665)]

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
for ax, vals, cis, title in (
    (axes[0], em, em_ci, "Exact match (answer set)"),
    (axes[1], f1, f1_ci, "F1 (answer set)"),
):
    y = list(range(len(arms)))[::-1]
    for yi, v, (lo, hi) in zip(y, vals, cis):
        ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color="black",
                    ecolor="black", elinewidth=1.2, capsize=4, markersize=5)
        ax.text(hi + 0.03, yi, f"{v:.3f}", va="center", ha="left", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(arms)
    ax.set_xlim(0, 1.08)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("paper2-fig1-em-f1-2026-11-21.png", dpi=200)
plt.close(fig)

# ---------- Figure 2: paired EM differences ----------
rows = [
    ("vs oracle",        -0.070, -0.094, -0.048),
    ("vs always-multi",   0.026, -0.008,  0.060),
    ("vs always-single",  0.264,  0.222,  0.306),
]
fig, ax = plt.subplots(figsize=(7.0, 3.2))
y = [2, 1, 0]
for yi, (name, v, lo, hi) in zip(y, rows):
    ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], fmt="o", color="black",
                ecolor="black", elinewidth=1.2, capsize=4, markersize=5)
    ax.text(v, yi + 0.32, f"{v:+.3f}", va="bottom", ha="center", fontsize=10)
ax.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlim(-0.13, 0.36)
ax.set_ylim(-0.55, 2.75)
ax.set_xlabel("Difference in exact match")
ax.set_title("Learned router minus other arm, exact match (paired, 95% bootstrap CI)", fontsize=10.5)
ax.grid(axis="x", color="#dddddd", linewidth=0.7)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("paper2-fig2-paired-diff-2026-11-21.png", dpi=200)
plt.close(fig)
print("figures written")
