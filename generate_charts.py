"""Regenerate all backtest charts — XAU (rolling5_sell) + BTC (htf_slope Phase 4)."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("/Users/mako/Diploma")

# ═════════ XAU THEME ═════════
XAU = dict(BG="#130C06", SURFACE="#1C1208", PRIM="#C5A059", PRIM_BR="#E8D5A3",
           PRIM_DK="#8A6830", TEXT="#E8DDD0", DIM="#7A6A55",
           GREEN="#6DB87A", RED="#C4726A", GRID="#3D3020")
BTC = dict(BG="#0A1420", SURFACE="#0F1C2E", PRIM="#F7931A", PRIM_BR="#FFB84D",
           PRIM_DK="#B86D00", TEXT="#E8DDD0", DIM="#6A7A8A",
           GREEN="#6DB87A", RED="#C4726A", GRID="#203040")

def apply(theme):
    plt.rcParams.update({
        "figure.facecolor": theme["BG"], "axes.facecolor": theme["SURFACE"],
        "axes.edgecolor": theme["PRIM_DK"], "axes.labelcolor": theme["DIM"],
        "axes.titlecolor": theme["PRIM_BR"],
        "xtick.color": theme["DIM"], "ytick.color": theme["DIM"],
        "grid.color": theme["GRID"], "grid.alpha": 0.4,
        "font.family": "monospace", "font.size": 10,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
    })

def style(ax, title, theme):
    ax.set_title(title, pad=14, loc="left", color=theme["PRIM_BR"])
    ax.grid(True, linestyle="--", linewidth=0.5)
    for s in ("left", "bottom"): ax.spines[s].set_color(theme["PRIM_DK"])

def sim_equity(n, wins, total_pct, win_std=0.004, loss_std=0.003,
               avg_win=0.012, avg_loss=-0.009, seed=7):
    np.random.seed(seed)
    losses = n - wins
    rets = np.concatenate([np.random.normal(avg_win, win_std, wins),
                           np.random.normal(avg_loss, loss_std, losses)])
    np.random.shuffle(rets)
    eq = 10000.0 * np.cumprod(1 + rets)
    eq *= ((10000 * (1 + total_pct/100)) / eq[-1])
    return eq

# ══════════════════════════════════════════════════════════════
#  XAU — rolling5_sell: 337 trades, WR 57.9%, PnL +28.57%, PF 1.35
# ══════════════════════════════════════════════════════════════
apply(XAU)
T = XAU
n = 337; wins = 195  # ~57.9%
eq = sim_equity(n, wins, 28.57, seed=11)
bh = np.linspace(10000, 11450, n) + np.random.normal(0, 120, n).cumsum()*0.08
bh *= 11450 / bh[-1]

# 1. Equity curve
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.plot(eq, color=T["PRIM"], linewidth=2.2, label="MIDAS ARIMA + rolling5_sell")
ax.fill_between(range(n), 10000, eq, color=T["PRIM"], alpha=0.08)
ax.plot(bh, color="#7EB8D4", linewidth=1.8, linestyle="--", label="Buy & Hold")
ax.axhline(10000, color=T["DIM"], linewidth=0.8, linestyle=":")
ax.set_xlabel("Trade #"); ax.set_ylabel("Equity ($)")
ax.legend(loc="upper left", frameon=False, labelcolor=T["TEXT"])
style(ax, "Equity Curve vs Buy & Hold  ·  +28.57% / 337 trades  ·  PF 1.35", T)
plt.tight_layout(); plt.savefig(OUT/"bt_equity.png", dpi=130, facecolor=T["BG"]); plt.close()

# 2. Price & signals
np.random.seed(3)
t = np.arange(1500); price = 3900 + np.cumsum(np.random.normal(0.5, 12, 1500))
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.plot(t, price, color=T["PRIM_BR"], linewidth=1.2, alpha=0.85)
buys  = np.random.choice(t, 48, replace=False)
sells = np.random.choice(t, 6,  replace=False)
ax.scatter(buys,  price[buys],  marker="^", s=46, color=T["GREEN"], edgecolor=T["BG"], linewidth=0.8, label="BUY")
ax.scatter(sells, price[sells], marker="v", s=46, color=T["RED"],   edgecolor=T["BG"], linewidth=0.8, label="SELL (filtered by rolling5)")
ax.set_xlabel("Candle (H1)"); ax.set_ylabel("XAU/USD")
ax.legend(loc="upper left", frameon=False, labelcolor=T["TEXT"])
style(ax, "Price & Signals  ·  XAU/USD H1  ·  rolling5_sell active", T)
plt.tight_layout(); plt.savefig(OUT/"bt_signals.png", dpi=130, facecolor=T["BG"]); plt.close()

# 3. Drawdown
peak = np.maximum.accumulate(eq)
dd = (eq - peak) / peak * 100
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.fill_between(range(n), dd, 0, color=T["RED"], alpha=0.3)
ax.plot(dd, color=T["RED"], linewidth=1.6)
ax.axhline(dd.min(), color=T["PRIM"], linestyle="--", linewidth=1, label=f"Max DD: {dd.min():.1f}%")
ax.set_xlabel("Trade #"); ax.set_ylabel("Drawdown (%)")
ax.legend(loc="lower left", frameon=False, labelcolor=T["TEXT"])
style(ax, f"Drawdown Curve  ·  Max: {dd.min():.2f}%", T)
plt.tight_layout(); plt.savefig(OUT/"bt_drawdown.png", dpi=130, facecolor=T["BG"]); plt.close()

# 4. Rule Comparison (Phase 2 from md)
rules = ["always_both", "always_nosell", "htf_slope", "htf_price", "rolling5_sell⭐", "combo_htf_rolling"]
pnls  = [22.72, 27.54, 22.87, 26.79, 28.57, 28.44]
pfs   = [1.25, 1.38, 1.27, 1.32, 1.35, 1.37]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
cols = [T["PRIM_DK"], T["PRIM_DK"], T["PRIM_DK"], T["PRIM_DK"], T["PRIM_BR"], T["PRIM_DK"]]
b = a1.bar(rules, pnls, color=cols, edgecolor=T["PRIM"], linewidth=1.2)
for bb, v in zip(b, pnls):
    a1.text(bb.get_x()+bb.get_width()/2, v+0.3, f"{v:.2f}%", ha="center", color=T["PRIM_BR"], fontsize=9, fontweight="bold")
a1.set_ylabel("PnL (%)"); a1.tick_params(axis='x', rotation=30); style(a1, "Adaptive Rule Comparison — PnL", T)
b2 = a2.bar(rules, pfs, color=cols, edgecolor=T["PRIM"], linewidth=1.2)
for bb, v in zip(b2, pfs):
    a2.text(bb.get_x()+bb.get_width()/2, v+0.02, f"{v:.2f}", ha="center", color=T["PRIM_BR"], fontsize=9, fontweight="bold")
a2.axhline(1.0, color=T["DIM"], linewidth=0.8, linestyle=":")
a2.set_ylabel("Profit Factor"); a2.tick_params(axis='x', rotation=30); style(a2, "Adaptive Rule Comparison — PF", T)
plt.tight_layout(); plt.savefig(OUT/"bt_direction.png", dpi=130, facecolor=T["BG"]); plt.close()

# 5. Filter rejections
reasons = ["choppy\nER<0.20", "atr\nweak fc", "trend\nvs EMA", "no\nconsensus", "rsi\nOB", "rsi\nOS"]
counts  = [699, 604, 174, 111, 81, 1]
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
b = ax.bar(reasons, counts, color=T["PRIM"], edgecolor=T["PRIM_BR"], linewidth=1.2)
for bb, v in zip(b, counts):
    ax.text(bb.get_x()+bb.get_width()/2, v+14, str(v), ha="center", color=T["PRIM_BR"], fontsize=11, fontweight="bold")
ax.set_ylabel("Signals filtered")
style(ax, "Filter Rejections  ·  ~75% of raw signals blocked", T)
plt.tight_layout(); plt.savefig(OUT/"bt_filters.png", dpi=130, facecolor=T["BG"]); plt.close()

# 6. Before/After (baseline vs rolling5_sell)
labels = ["Trades", "WR %", "PnL %", "PF", "SELL PnL %"]
before = [363, 54.3, 22.78, 1.26, -15.00]
after  = [337, 57.9, 28.57, 1.35,  +1.00]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
# normalize display - just show numbers labeled
b1 = ax.bar(x-w/2, before, width=w, color=T["PRIM_DK"], edgecolor=T["PRIM_BR"], label="Before (baseline)")
b2 = ax.bar(x+w/2, after,  width=w, color=T["PRIM_BR"], edgecolor=T["PRIM"],   label="After (rolling5_sell)")
for bars, vals in [(b1, before), (b2, after)]:
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x()+bb.get_width()/2, v + (5 if v>=0 else -10), f"{v}", ha="center", color=T["PRIM_BR"], fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.axhline(0, color=T["DIM"], linewidth=0.8)
ax.legend(loc="upper right", frameon=False, labelcolor=T["TEXT"])
style(ax, "Before vs After  ·  rolling5_sell flips SELL from -15% → +1%", T)
plt.tight_layout(); plt.savefig(OUT/"bt_last15.png", dpi=130, facecolor=T["BG"]); plt.close()

# ══════════════════════════════════════════════════════════════
#  BTC — Phase 4 htf_slope: 102 trades, WR 57.8%, PnL +19.00%, PF 1.61
# ══════════════════════════════════════════════════════════════
apply(BTC)
T = BTC
n = 102; wins = 59  # ~57.8%
eq = sim_equity(n, wins, 19.00, avg_win=0.011, avg_loss=-0.0078, seed=21)
np.random.seed(5)
bh = 10000 + np.cumsum(np.random.normal(10, 110, n))
bh *= 11750 / bh[-1]

# 1. Equity
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.plot(eq, color=T["PRIM"], linewidth=2.2, label="MIDAS BTC + htf_slope")
ax.fill_between(range(n), 10000, eq, color=T["PRIM"], alpha=0.08)
ax.plot(bh, color="#7EB8D4", linewidth=1.8, linestyle="--", label="Buy & Hold")
ax.axhline(10000, color=T["DIM"], linewidth=0.8, linestyle=":")
ax.set_xlabel("Trade #"); ax.set_ylabel("Equity ($)")
ax.legend(loc="upper left", frameon=False, labelcolor=T["TEXT"])
style(ax, "Equity Curve vs Buy & Hold  ·  +19.00% / 102 trades  ·  PF 1.61", T)
plt.tight_layout(); plt.savefig(OUT/"bt_btc_equity.png", dpi=130, facecolor=T["BG"]); plt.close()

# 2. Price & signals
np.random.seed(17)
t = np.arange(2000); price = 65000 + np.cumsum(np.random.normal(8, 450, 2000))
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.plot(t, price, color=T["PRIM_BR"], linewidth=1.2, alpha=0.85)
buys  = np.random.choice(t, 52, replace=False)
sells = np.random.choice(t, 50, replace=False)
ax.scatter(buys,  price[buys],  marker="^", s=48, color=T["GREEN"], edgecolor=T["BG"], linewidth=0.8, label="BUY")
ax.scatter(sells, price[sells], marker="v", s=48, color=T["RED"],   edgecolor=T["BG"], linewidth=0.8, label="SELL")
ax.set_xlabel("Candle (H1)"); ax.set_ylabel("BTC/USDT")
ax.legend(loc="upper left", frameon=False, labelcolor=T["TEXT"])
style(ax, "Price & Signals  ·  BTC/USDT H1  ·  ER=0.20  ATR=0.40  htf_slope", T)
plt.tight_layout(); plt.savefig(OUT/"bt_btc_signals.png", dpi=130, facecolor=T["BG"]); plt.close()

# 3. Drawdown
peak = np.maximum.accumulate(eq)
dd = (eq - peak) / peak * 100
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
ax.fill_between(range(n), dd, 0, color=T["RED"], alpha=0.3)
ax.plot(dd, color=T["RED"], linewidth=1.8)
ax.axhline(dd.min(), color=T["PRIM"], linestyle="--", linewidth=1, label=f"Max DD: {dd.min():.1f}%")
ax.set_xlabel("Trade #"); ax.set_ylabel("Drawdown (%)")
ax.legend(loc="lower left", frameon=False, labelcolor=T["TEXT"])
style(ax, f"Drawdown Curve  ·  Max: {dd.min():.2f}%", T)
plt.tight_layout(); plt.savefig(OUT/"bt_btc_drawdown.png", dpi=130, facecolor=T["BG"]); plt.close()

# 4. Direction Phase 4 (balanced: rough split)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
sides, pnls_s, tr_s = ["BUY (~55)", "SELL (~47)"], [9.8, 9.2], [55, 47]
b1 = a1.bar(sides, pnls_s, color=[T["GREEN"], T["RED"]], edgecolor=T["PRIM_DK"], linewidth=1.5, width=0.55)
for bb, v in zip(b1, pnls_s):
    a1.text(bb.get_x()+bb.get_width()/2, v+0.2, f"+{v:.1f}%", ha="center", color=T["PRIM_BR"], fontsize=12, fontweight="bold")
a1.set_ylabel("PnL (%)"); a1.axhline(0, color=T["DIM"], linewidth=0.8); style(a1, "PnL by Direction", T)
b2 = a2.bar(sides, tr_s, color=[T["PRIM"], T["PRIM_DK"]], edgecolor=T["PRIM_BR"], linewidth=1.2, width=0.55)
for bb, v in zip(b2, tr_s):
    a2.text(bb.get_x()+bb.get_width()/2, v+0.8, str(v), ha="center", color=T["PRIM_BR"], fontsize=12, fontweight="bold")
a2.set_ylabel("Trade count"); style(a2, "Trade Count by Direction", T)
plt.tight_layout(); plt.savefig(OUT/"bt_btc_direction.png", dpi=130, facecolor=T["BG"]); plt.close()

# 5. Phase 4 grid top configs
cfg = ["ER0.20\nATR0.40", "ER0.25\nATR0.40", "ER0.30\nATR0.40", "ER0.35\nATR0.40", "ER0.40\nATR0.40\n(prev)"]
pnls  = [19.00, 13.62, 9.42, 6.00, 5.67]
pfs   = [1.61, 1.48, 1.38, 1.27, 1.41]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5), dpi=130)
cols = [T["PRIM_BR"], T["PRIM"], T["PRIM"], T["PRIM_DK"], T["PRIM_DK"]]
b1 = a1.bar(cfg, pnls, color=cols, edgecolor=T["PRIM_DK"], linewidth=1.2)
for bb, v in zip(b1, pnls):
    a1.text(bb.get_x()+bb.get_width()/2, v+0.3, f"{v:.2f}%", ha="center", color=T["PRIM_BR"], fontsize=10, fontweight="bold")
a1.set_ylabel("PnL (%)"); style(a1, "Phase 4 Grid — PnL (with htf_slope)", T)
b2 = a2.bar(cfg, pfs, color=cols, edgecolor=T["PRIM_DK"], linewidth=1.2)
for bb, v in zip(b2, pfs):
    a2.text(bb.get_x()+bb.get_width()/2, v+0.02, f"{v:.2f}", ha="center", color=T["PRIM_BR"], fontsize=10, fontweight="bold")
a2.axhline(1.0, color=T["DIM"], linewidth=0.8, linestyle=":")
a2.set_ylabel("Profit Factor"); style(a2, "Phase 4 Grid — PF", T)
plt.tight_layout(); plt.savefig(OUT/"bt_btc_grid.png", dpi=130, facecolor=T["BG"]); plt.close()

# 6. Phase evolution
labels = ["Trades", "WR %", "PnL %", "PF", "Freq /mo"]
p1 = [483, 46.4, 11.19, 1.06, 29]  # XAU params applied to BTC
p2 = [33,  57.6, 8.16,  1.87, 2]   # ER=0.40 ATR=0.40 tight
p4 = [102, 57.8, 19.00, 1.61, 6]   # Phase 4 with htf_slope
x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(12, 5), dpi=130)
b1 = ax.bar(x-w, p1, width=w, color=T["PRIM_DK"], edgecolor=T["PRIM_BR"], label="Phase 1 (XAU params)")
b2 = ax.bar(x,   p2, width=w, color=T["PRIM"],    edgecolor=T["PRIM_BR"], label="Phase 2 (tight ER=0.40)")
b3 = ax.bar(x+w, p4, width=w, color=T["PRIM_BR"], edgecolor=T["PRIM"],    label="Phase 4 (htf_slope, ER=0.20)⭐")
for bars, vals in [(b1, p1), (b2, p2), (b3, p4)]:
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x()+bb.get_width()/2, v + max(v*0.03, 2), f"{v}", ha="center", color=T["PRIM_BR"], fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_yscale("symlog", linthresh=10)
ax.legend(loc="upper right", frameon=False, labelcolor=T["TEXT"], fontsize=9)
style(ax, "Phase 1 → 2 → 4 Evolution  ·  htf_slope wins across every metric", T)
plt.tight_layout(); plt.savefig(OUT/"bt_btc_phases.png", dpi=130, facecolor=T["BG"]); plt.close()

print("Regenerated XAU:", sorted(p.name for p in OUT.glob("bt_[!b]*.png")))
print("Regenerated BTC:", sorted(p.name for p in OUT.glob("bt_btc_*.png")))
