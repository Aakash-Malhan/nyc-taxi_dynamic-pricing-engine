import pandas as pd
import numpy as np
import joblib
import gradio as gr
from pathlib import Path
from datetime import datetime

# ---------------- Paths ----------------
DATA_DIR = Path("data")
ZONE_HOUR_FEATS = DATA_DIR / "zone_hour_features.parquet"
ZONE_HOUR_AGG   = DATA_DIR / "zone_hour_agg.parquet"
BASEFARE        = DATA_DIR / "zone_hour_basefare.parquet"
MODEL_PATH      = DATA_DIR / "zone_hour_surge_xgb.joblib"
# Optional (if present, we show zone names)
ZONE_LOOKUP_CSV = DATA_DIR / "taxi_zone_lookup.csv"

needed = [ZONE_HOUR_AGG, BASEFARE, ZONE_HOUR_FEATS, MODEL_PATH]
missing = [p.name for p in needed if not p.exists()]
if missing:
    raise FileNotFoundError(
        f"Missing: {missing}. Put all artifacts in ./data/ "
        "(zone_hour_agg.parquet, zone_hour_basefare.parquet, "
        "zone_hour_features.parquet, zone_hour_surge_xgb.joblib)."
    )

# ---------------- Load artifacts ----------------
agg = pd.read_parquet(ZONE_HOUR_AGG)
base = pd.read_parquet(BASEFARE)
feats_ref = pd.read_parquet(ZONE_HOUR_FEATS)
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_cols = bundle["feature_cols"]  # ['hour','dow','month','is_weekend','pulocationid','demand','supply','gap']

# Optional zone names (for prettier dropdown labels)
zone_names = None
if ZONE_LOOKUP_CSV.exists():
    z = pd.read_csv(ZONE_LOOKUP_CSV)
    z = z.rename(columns={"LocationID": "locationid", "Borough": "borough", "Zone": "zone"})
    zone_names = z[["locationid", "borough", "zone"]].copy()

# ---------------- Vectorized helpers (work for scalars & arrays) ----------------
def is_rush(h):
    """Return 1 if hour in rush (7–10 or 16–19). Works for scalar or array-like."""
    h_arr = np.asarray(h)
    rush = ((h_arr >= 7) & (h_arr <= 10)) | ((h_arr >= 16) & (h_arr <= 19))
    rush = rush.astype(int)
    return int(rush.item()) if np.isscalar(h) else rush

def weekend_flag(dow):
    """Training used DuckDB DOW: 0=Sun .. 6=Sat."""
    dow_arr = np.asarray(dow)
    wknd = ((dow_arr == 0) | (dow_arr == 6)).astype(int)
    return int(wknd.item()) if np.isscalar(dow) else wknd

def pressure_x(gap, supply, hour):
    """(gap/(supply+1)) + 0.5 * is_rush; vectorized & scalar-safe."""
    gap_a    = np.asarray(gap, dtype=float)
    supply_a = np.asarray(supply, dtype=float)
    hour_a   = np.asarray(hour)
    return (gap_a / (supply_a + 1.0)) + 0.5 * is_rush(hour_a)

def driver_bonus_from_x(x):
    x_a = np.asarray(x, dtype=float)
    bonus = np.where(x_a > 0.5, 2.0 + 3.0 * (1.0 / (1.0 + np.exp(-x_a))), 0.0)
    return float(bonus.item()) if np.isscalar(x) else bonus

def default_time():
    now = datetime.utcnow()
    py_dow = now.weekday()        # Mon=0..Sun=6
    duck_dow = (py_dow + 1) % 7   # Sun=0..Sat=6  (matches training)
    return now.hour, duck_dow, now.month

# ---------------- Zone dropdown (labels -> ids) ----------------
def zones_list():
    ids = sorted(agg["pulocationid"].unique().tolist())
    if zone_names is None:
        labels = [str(i) for i in ids]
        label_to_id = {str(i): str(i) for i in ids}
        return labels, label_to_id
    m = {int(r.locationid): f"{r.zone} — {r.borough} ({int(r.locationid)})" for _, r in zone_names.iterrows()}
    labels = [m.get(i, str(i)) for i in ids]
    label_to_id = {m.get(i, str(i)): str(i) for i in ids}
    return labels, label_to_id

labels, label_to_id = zones_list()

# ---------------- Explanations ----------------
STATIC_EXPLAIN = """
### What’s happening under the hood
**Inputs →** (pickup *zone*, *hour*, *day-of-week*, *month*, and demand/supply).  
**Features →** We compute **demand**, **supply** and **gap** at the *zone×hour* level from real TLC trips (all months you trained on).  
**Surge model →** An XGBoost regressor predicts **surge_multiplier** from `{hour, dow (0=Sun..6=Sat), month, is_weekend, zone-id, demand, supply, gap}`.  
**Base fare →** Median historical **fare_amount** for the same zone and **hour-of-day** (fallback to zone median).  
**Pricing →** `recommended_price = base_fare × surge_pred`, clipped to a reasonable range.  
**Driver bonus →** A rule based on the **pressure metric** `x = gap/(supply+1) + 0.5·rush_hour`; suggest a bonus only if pressure is high.  
**Acceptance simulation →** A simple curve estimates how acceptance shifts when price changes, to illustrate the **revenue vs. acceptance** trade-off.
This is the same pattern used in real marketplaces: aggregate to the decision unit (zone×time), predict pressure, price accordingly, and optionally add incentives.
"""

def exec_summary(zone_label, hour, dow, month, demand, supply, gap, surge, base_fare, rec_price, bonus, p_base, p_ml, source_info):
    sign = "higher" if rec_price >= base_fare else "lower"
    pct  = 0.0 if base_fare == 0 else (rec_price - base_fare) / (base_fare + 1e-6) * 100
    return (
        "### Executive summary\n"
        f"- **Context:** Zone **{zone_label}**, Hour **{hour}**, DOW (0=Sun) **{dow}**, Month **{month}** "
        f"(demand **{demand}**, supply **{supply}**, gap **{gap}**; src: {source_info}).\n"
        f"- **Model:** Predicted **surge ≈ {surge:.2f}**; base fare **${base_fare:.2f}**.\n"
        f"- **Recommendation:** **Price ${rec_price:.2f}** ({sign} by {pct:+.1f}% vs. base); **Driver bonus ${bonus:.2f}**.\n"
        f"- **Impact:** Expected acceptance baseline **{p_base:.2f}** → ML **{p_ml:.2f}** (illustrative), "
        "trading some acceptance for higher per-trip revenue when pressure is high."
    )

# ---------------- Core inference ----------------
def recommend(puloc_label, hour, dow, month, demand_input, supply_input, use_latest_from_data):
    pulocationid = int(label_to_id[str(puloc_label)])

    # Demand/supply source
    if use_latest_from_data:
        zrows = agg[agg["pulocationid"] == pulocationid]
        if not zrows.empty:
            latest = zrows.sort_values("pickup_hour").iloc[-1]
            demand = int(latest["demand"])
            supply = int(latest["supply"])
            gap = int(latest["gap"])
            source_info = f"latest in data ({latest['pickup_hour']})"
        else:
            demand, supply, gap = 0, 0, 0
            source_info = "no history; using zeros"
    else:
        demand = int(demand_input)
        supply = int(supply_input)
        gap = demand - supply
        source_info = "manual entry"

    # Feature row for surge prediction
    X = pd.DataFrame([{
        "hour": int(hour),
        "dow": int(dow),                      # 0=Sun .. 6=Sat
        "month": int(month),
        "is_weekend": weekend_flag(int(dow)),
        "pulocationid": pulocationid,
        "demand": demand,
        "supply": supply,
        "gap": gap,
    }])[feature_cols].astype({
        "hour":"int16","dow":"int16","month":"int16","is_weekend":"int8",
        "pulocationid":"int32","demand":"int32","supply":"int32","gap":"int32"
    })

    surge_pred = float(model.predict(X)[0])

    # Base fare lookup: hour-of-day for the zone; else zone median; else fallback
    bf_zone = base[base["pulocationid"] == pulocationid]
    if not bf_zone.empty:
        bf_zone = bf_zone.assign(hod=bf_zone["pickup_hour"].dt.hour.astype(int))
        bf_match = bf_zone[bf_zone["hod"] == int(hour)]
        base_fare = float(bf_match["base_fare"].median()) if not bf_match.empty else float(bf_zone["base_fare"].median())
    else:
        base_fare = 12.0

    # Price & bonus
    recommended_price = float(np.clip(base_fare * surge_pred, 5.0, 150.0))
    x = pressure_x(gap, supply, int(hour))
    bonus = float(driver_bonus_from_x(x))

    # Acceptance summary (illustrative curve)
    def accept_prob(price, base):
        rel = (price - base) / (base + 1e-6)
        p = 0.95 - 0.7 * (1 / (1 + np.exp(-4 * rel)))
        return float(np.clip(p, 0.05, 0.95))

    p_acc_base = accept_prob(base_fare, base_fare)
    p_acc_ml   = accept_prob(recommended_price, base_fare)

    pretty_zone = puloc_label

    # Tabular output
    out = pd.DataFrame([{
        "pulocationid": int(pulocationid),
        "hour": int(hour),
        "dow(0=Sun)": int(dow),
        "month": int(month),
        "demand": int(demand),
        "supply": int(supply),
        "gap": int(gap),
        "surge_pred": round(surge_pred, 3),
        "base_fare": round(base_fare, 2),
        "recommended_price": round(recommended_price, 2),
        "driver_bonus": round(bonus, 2),
        "p_acc_base": round(p_acc_base, 3),
        "p_acc_ml": round(p_acc_ml, 3),
    }])

    # Plain-English one-liner & detailed executive summary
    summary = (
        f"Zone: {pretty_zone} | Hour: {hour} | DOW (0=Sun): {dow} | Month: {month} | "
        f"Source: {source_info} | Demand={demand} | Supply={supply} | Gap={gap} → "
        f"Surge ≈ {surge_pred:.2f}; Base ${base_fare:.2f} ⇒ **Price ${recommended_price:.2f}**; "
        f"Bonus ${bonus:.2f}; Acceptance baseline {p_acc_base:.2f} → ML {p_acc_ml:.2f}"
    )

    exec_md = exec_summary(
        zone_label=pretty_zone, hour=hour, dow=dow, month=month,
        demand=demand, supply=supply, gap=gap,
        surge=surge_pred, base_fare=base_fare, rec_price=recommended_price,
        bonus=bonus, p_base=p_acc_base, p_ml=p_acc_ml, source_info=source_info
    )

    return out, summary, STATIC_EXPLAIN, exec_md

# ---------------- Hot zones panel ----------------
def hot_zones_table(k=5):
    tmp = agg.copy()
    tmp["hour"] = tmp["pickup_hour"].dt.hour.astype(int)
    tmp["x"] = pressure_x(tmp["gap"], tmp["supply"], tmp["hour"])  # vectorized
    res = (tmp.groupby("pulocationid")["x"]
             .mean()
             .sort_values(ascending=False)
             .head(k)
             .reset_index())

    if zone_names is not None:
        res = res.merge(zone_names, left_on="pulocationid", right_on="locationid", how="left")
        res["zone_label"] = res["zone"].fillna(res["pulocationid"].astype(str)) + " — " + res["borough"].fillna("")
        res = res[["zone_label","x"]].rename(columns={"x":"avg_pressure"})
    else:
        res = res.rename(columns={"pulocationid":"zone_id", "x":"avg_pressure"})
    return res

# ---------------- UI ----------------
def_time = default_time()
labels_sorted = sorted(labels)

with gr.Blocks(title="NYC Pricing & Incentive Recommender") as demo:
    gr.Markdown(
        "# NYC Pricing & Incentive Recommender\n"
        "Predict **surge**, **price**, and **driver bonus** by pickup **zone × hour**.\n"
        "_Note: DOW uses 0=Sunday .. 6=Saturday (matches model training)._"
    )

    with gr.Row():
        puloc_dd = gr.Dropdown(choices=labels_sorted, value=labels_sorted[0], label="Pickup zone")
        hour_in  = gr.Slider(0, 23, value=def_time[0], step=1, label="Hour of day")
        dow_in   = gr.Slider(0, 6,  value=def_time[1], step=1, label="Day of week (0=Sun)")
        month_in = gr.Slider(1, 12, value=def_time[2], step=1, label="Month")

    use_latest = gr.Checkbox(value=True, label="Use latest demand/supply from data for this zone")
    with gr.Row():
        demand_in = gr.Number(value=20, precision=0, label="Demand (manual if unchecked)")
        supply_in = gr.Number(value=15, precision=0, label="Supply (manual if unchecked)")

    run_btn = gr.Button("Recommend")

    out_df   = gr.Dataframe(label="Recommendation", wrap=True)
    out_txt  = gr.Markdown()
    explain_md = gr.Markdown()        # << What’s happening
    exec_md    = gr.Markdown()        # << Executive summary

    gr.Markdown("### Top pressure zones (dataset average)")
    hot_df = gr.Dataframe(hot_zones_table(), wrap=True)

    run_btn.click(
        recommend,
        inputs=[puloc_dd, hour_in, dow_in, month_in, demand_in, supply_in, use_latest],
        outputs=[out_df, out_txt, explain_md, exec_md]
    )

if __name__ == "__main__":
    demo.launch()
