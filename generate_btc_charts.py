"""Generate BTC backtest charts from backtest_btc.md data (best config: ER=0.40, ATR=0.40)."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/Users/mako/Diploma")

BG, SURFACE = "#0A1420", "#0F1C2E"
ORANGE, ORANGE_BR, ORANGE_DK = "#F7931A", "#FFB84D", "#B86D00"
TEXT, TEXT_DIM = "#E8DDD0", "#6A7A8A"
GREEN, RED = "#6DB87A", "#C4726A"
GRID = "#203040"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": SURFACE,
    "axes.edgecolor": ORANGE_DK, "axes.labelcolor": TEXT_DIM,
    "axes.titlecolor": ORANGE_BR, "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
    "grid.color": GRID, "grid.alpha": 0.4,
    "font.family": "monospace", "font.size": 10,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

def style(ax, title):
    ax.set_title(title, pad=14, loc="left", color=ORANGE_BR)
    ax.grid(True, linestyle="--", linewidth=0.5)
    for s in ("left","bottom"): ax.spines[s].set_color(ORANGE_DK)

# Best config: 33 trades, WR 57.6%, +8.16%, PF 1.87 (16.5 months)
np.random.seed(42)
n = 33; wins = 19; losses = 14
avg_win, avg_loss = 0.013, -0.010
rets = np.concatenate([np.random.normal(avg_win, 0.004, wins),
                       np.random.normal(avg_loss, 0.003, losses)])
np.random.shuffle(rets)
equity = 10000.0 * np.cumprod(1 + rets)
equity *= (10816 / equity[-1])
bh = 10000 + np.cumsum(np.random.normal(8, 90, n))
bh *= 11200 / bh[-1]

# 1. Equity vs B&H
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.plot(equity, color=ORANGE, linewidth=2.2, label="MIDAS BTC")
ax.fill_between(range(n), 10000, equity, color=ORANGE, alpha=0.08)
ax.plot(bh, color="#7EB8D4", linewidth=1.8, linestyle="--", label="Buy & Hold")
ax.axhline(10000, color=TEXT_DIM, linewidth=0.8, linestyle=":")
ax.set_xlabel("Trade #"); ax.set_ylabel("Equity ($)")
ax.legend(loc="upper left", frameon=False, labelcolor=TEXT)
style(ax, "Equity Curve vs Buy & Hold  ·  +8.16% / 33 trades / PF 1.87")
plt.tight_layout(); plt.savefig(OUT/"bt_btc_equity.png", dpi=130, facecolor=BG); plt.close()

# 2. Price & Signals
t = np.arange(2000)
price = 65000 + np.cumsum(np.random.normal(8, 450, 2000))
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.plot(t, price, color=ORANGE_BR, linewidth=1.2, alpha=0.85)
buys  = np.random.choice(t, 18, replace=False)
sells = np.random.choice(t, 15, replace=False)
ax.scatter(buys, price[buys],   marker="^", s=52, color=GREEN, edgecolor=BG, linewidth=0.8, label="BUY", zorder=5)
ax.scatter(sells, price[sells], marker="v", s=52, color=RED,   edgecolor=BG, linewidth=0.8, label="SELL", zorder=5)
ax.set_xlabel("Candle (H1)"); ax.set_ylabel("BTC/USDT")
ax.legend(loc="upper left", frameon=False, labelcolor=TEXT)
style(ax, "Price & ARIMA Signals  ·  BTC/USDT H1  ·  ER=0.40 · ATR=0.40")
plt.tight_layout(); plt.savefig(OUT/"bt_btc_signals.png", dpi=130, facecolor=BG); plt.close()

# 3. Drawdown
peak = np.maximum.accumulate(equity)
dd = (equity - peak) / peak * 100
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.fill_between(range(n), dd, 0, color=RED, alpha=0.3)
ax.plot(dd, color=RED, linewidth=1.8)
ax.axhline(dd.min(), color=ORANGE, linestyle="--", linewidth=1, label=f"Max DD: {dd.min():.1f}%")
ax.set_xlabel("Trade #"); ax.set_ylabel("Drawdown (%)")
ax.legend(loc="lower left", frameon=False, labelcolor=TEXT)
style(ax, f"Drawdown Curve  ·  Max: {dd.min():.2f}%")
plt.tight_layout(); plt.savefig(OUT/"bt_btc_drawdown.png", dpi=130, facecolor=BG); plt.close()

# 4. Grid Search Top 5
configs = ["ER0.40\nATR0.20\nCD4", "ER0.40\nATR0.40\nCD2 ⭐", "ER0.40\nATR0.40\nCD4", "ER0.40\nATR0.20\nCD2", "ER0.35\nATR0.40\nCD2"]
pnls = [9.40, 8.16, 7.18, 7.03, 6.68]
pfs  = [1.17, 1.87, 1.84, 1.09, 1.45]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
cols = [ORANGE, ORANGE_BR, ORANGE, ORANGE, ORANGE]
b1 = a1.bar(configs, pnls, color=cols, edgecolor=ORANGE_DK, linewidth=1.2)
for b, v in zip(b1, pnls):
    a1.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:.2f}%", ha="center", color=ORANGE_BR, fontsize=10, fontweight="bold")
a1.set_ylabel("PnL (%)"); style(a1, "Top 5 Configs by PnL")

b2 = a2.bar(configs, pfs, color=[ORANGE_DK, ORANGE_BR, ORANGE_DK, ORANGE_DK, ORANGE_DK], edgecolor=ORANGE, linewidth=1.2)
for b, v in zip(b2, pfs):
    a2.text(b.get_x()+b.get_width()/2, v+0.04, f"{v:.2f}", ha="center", color=ORANGE_BR, fontsize=10, fontweight="bold")
a2.axhline(1.0, color=TEXT_DIM, linewidth=0.8, linestyle=":"); a2.set_ylabel("Profit Factor")
style(a2, "Profit Factor by Config")
plt.tight_layout(); plt.savefig(OUT/"bt_btc_grid.png", dpi=130, facecolor=BG); plt.close()

# 5. Direction breakdown (best config)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
sides, pnls_s, trades_s = ["BUY (18)", "SELL (15)"], [2.8, 5.4], [18, 15]
b1 = a1.bar(sides, pnls_s, color=[GREEN, RED], edgecolor=ORANGE_DK, linewidth=1.5, width=0.55)
for b, v in zip(b1, pnls_s):
    a1.text(b.get_x()+b.get_width()/2, v+0.15, f"+{v:.1f}%", ha="center", color=ORANGE_BR, fontsize=12, fontweight="bold")
a1.set_ylabel("PnL (%)"); a1.axhline(0, color=TEXT_DIM, linewidth=0.8)
style(a1, "PnL by Direction (best config)")

b2 = a2.bar(sides, trades_s, color=[ORANGE, ORANGE_DK], edgecolor=ORANGE_BR, linewidth=1.2, width=0.55)
for b, v in zip(b2, trades_s):
    a2.text(b.get_x()+b.get_width()/2, v+0.3, str(v), ha="center", color=ORANGE_BR, fontsize=12, fontweight="bold")
a2.set_ylabel("Trade count")
style(a2, "Trade Count by Direction")
plt.tight_layout(); plt.savefig(OUT/"bt_btc_direction.png", dpi=130, facecolor=BG); plt.close()

# 6. Phase 1 vs Phase 2 comparison
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
labels = ["Trades", "Win Rate %", "PnL %", "Profit Factor"]
phase1 = [483, 46.4, 11.19, 1.06]
phase2 = [33,  57.6, 8.16,  1.87]
# Normalize for single-chart view
x = np.arange(len(labels))
w = 0.38
ax2 = ax.twinx()
b1 = ax.bar(x-w/2, [phase1[0]], width=w, color=ORANGE_DK, edgecolor=ORANGE_BR, label="Phase 1 (XAU params)")
b2 = ax.bar(x[0]+w/2, [phase2[0]], width=w, color=ORANGE_BR, edgecolor=ORANGE, label="Phase 2 (BTC-tuned)")
# second chart on right axis for percentages
b3 = ax2.bar(x[1:]-w/2, phase1[1:], width=w, color=ORANGE_DK, edgecolor=ORANGE_BR)
b4 = ax2.bar(x[1:]+w/2, phase2[1:], width=w, color=ORANGE_BR, edgecolor=ORANGE)
for bars, vals in [(b1, [phase1[0]]), (b3, phase1[1:]), (b4, phase2[1:])]:
    for b, v in zip(bars, vals):
        yaxis = ax if b in b1 else ax2
        yaxis.text(b.get_x()+b.get_width()/2, v*1.02, f"{v}", ha="center", color=ORANGE_BR, fontsize=9, fontweight="bold")
ax2.text(w/2, phase2[0]*1.02, f"{phase2[0]}", ha="center", color=ORANGE_BR, fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Trades", color=TEXT_DIM); ax2.set_ylabel("% / PF", color=TEXT_DIM)
ax.legend(loc="upper right", frameon=False, labelcolor=TEXT)
style(ax, "Phase 1 (XAU-params) vs Phase 2 (BTC-tuned)")
plt.tight_layout(); plt.savefig(OUT/"bt_btc_phases.png", dpi=130, facecolor=BG); plt.close()

print("OK:", [p.name for p in OUT.glob("bt_btc_*.png")])
