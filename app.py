import io
import json
import sqlite3
import warnings
from datetime import datetime
from pathlib import Path

import folium
import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------- Paths ----------------
DATA_DIR = Path("data")
P = {
    "features": DATA_DIR / "zone_hour_features.parquet",
    "basefare": DATA_DIR / "zone_hour_basefare.parquet",
    "agg": DATA_DIR / "zone_hour_agg.parquet",
    "eta": DATA_DIR / "zone_hour_eta.parquet",
    "lookup": DATA_DIR / "taxi_zone_lookup.csv",
    "geojson": DATA_DIR / "taxi_zones.geojson",
    "model": DATA_DIR / "zone_hour_surge_xgb.joblib",  # <- correct name (no extra 's')
}

# ---------------- Load artifacts ----------------
features = pd.read_parquet(P["features"])
basefare = pd.read_parquet(P["basefare"])
agg = pd.read_parquet(P["agg"])
eta = pd.read_parquet(P["eta"]) if P["eta"].exists() else None
zones_gj = json.load(open(P["geojson"])) if P["geojson"].exists() else None

# Robust lookup loading (handles capitalization differences)
lk = pd.read_csv(P["lookup"])
cols = {c.lower(): c for c in lk.columns}
loc_col = cols.get("locationid") or cols.get("location_id")
zone_col = cols.get("zone")
boro_col = cols.get("borough")
if loc_col is None:
    raise ValueError("taxi_zone_lookup.csv must have 'LocationID' (or 'location_id').")

lk = lk.rename(columns={loc_col: "LocationID"})
if zone_col and boro_col:
    disp = lk[zone_col].astype(str) + " — " + lk[boro_col].astype(str)
elif zone_col:
    disp = lk[zone_col].astype(str)
elif boro_col:
    disp = lk[boro_col].astype(str)
else:
    disp = lk["LocationID"].astype(str)

lk["display"] = disp + " (" + lk["LocationID"].astype(int).astype(str) + ")"
# For dropdown (id,label)
zone_choices = sorted([(int(r.LocationID), r.display) for _, r in lk.iterrows()], key=lambda x: x[1])
# Quick id->label map
id_to_label = {i: label for i, label in zone_choices}

# Load model (bundle or raw)
mdl_obj = joblib.load(P["model"])
if isinstance(mdl_obj, dict) and "model" in mdl_obj:
    model = mdl_obj["model"]
    feature_cols = mdl_obj.get(
        "feature_cols",
        ["hour", "dow", "month", "is_weekend", "pulocationid", "demand", "supply", "gap"],
    )
else:
    model = mdl_obj
    feature_cols = ["hour", "dow", "month", "is_weekend", "pulocationid", "demand", "supply", "gap"]

# ---------------- SQLite logging ----------------
DB_PATH = DATA_DIR / "events.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events(
            ts TEXT,
            zone_label TEXT,
            pulocationid INTEGER,
            hour INTEGER,
            dow INTEGER,
            month INTEGER,
            demand INTEGER,
            supply INTEGER,
            gap INTEGER,
            surge_pred REAL,
            base_fare REAL,
            recommended_price REAL,
            driver_bonus REAL,
            p_acc_base REAL,
            p_acc_ml REAL
        )
        """
    )
    conn.commit()
    conn.close()
init_db()

def log_event(row: dict):
    try:
        conn = sqlite3.connect(DB_PATH)
        cols = ",".join(row.keys())
        qs = ",".join(["?"] * len(row))
        conn.execute(f"INSERT INTO events({cols}) VALUES({qs})", tuple(row.values()))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Log error:", e)

def export_events():
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM events ORDER BY ts DESC LIMIT 5000", conn)
        conn.close()
        out = DATA_DIR / "events_export.csv"
        df.to_csv(out, index=False)
        return str(out)
    except Exception:
        return None

# ---------------- Helpers ----------------
def is_weekend(dow):
    # DOW: 0=Sun .. 6=Sat
    return 1 if dow in (0, 6) else 0

def pressure_x(gap, supply, hour):
    # Simple pressure heuristic used for bonus & map
    return (gap / (supply + 1.0)) + 0.5 * (1 if (7 <= hour <= 10) or (16 <= hour <= 19) else 0)

def accept_prob(price, base):
    rel = (price - base) / (base + 1e-6)
    p = 0.95 - 0.7 * (1 / (1 + np.exp(-4 * rel)))
    return float(np.clip(p, 0.05, 0.95))

# ---------------- SHAP (robust) ----------------
def _make_shap_explainer(m):
    try:
        expl = shap.TreeExplainer(m, feature_names=feature_cols, model_output="raw")
        _ = expl(pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols), check_additivity=False)
        return expl, "tree"
    except Exception as e1:
        try:
            expl = shap.Explainer(m, feature_names=feature_cols)
            _ = expl(pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols))
            return expl, "generic"
        except Exception as e2:
            print("SHAP init failed:", repr(e1), "|", repr(e2))
            return None, "none"

shap_explainer, shap_mode = _make_shap_explainer(model)
print("SHAP mode:", shap_mode)

def shap_bar_np(feature_row: pd.DataFrame):
    try:
        if shap_explainer is not None:
            if shap_mode == "tree":
                sv = shap_explainer(feature_row, check_additivity=False)[0]
            else:
                sv = shap_explainer(feature_row)[0]
            vals = sv.values
            names = feature_row.columns
            dfp = pd.DataFrame({"feature": names, "shap": vals})
            dfp["abs"] = np.abs(dfp["shap"])
            dfp = dfp.sort_values("abs").tail(8)

            fig = plt.figure(figsize=(6, 4))
            plt.barh(dfp["feature"], dfp["shap"])
            plt.axvline(0, color="black", linewidth=0.8)
            plt.xlabel("SHAP contribution to surge")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return np.array(Image.open(buf).convert("RGB")).astype(np.uint8)
    except Exception as e:
        print("SHAP error:", e)

    # Fallback: feature_importances_
    try:
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_, dtype=float)
            dfp = pd.DataFrame({"feature": feature_cols, "score": fi}).sort_values("score").tail(8)
            fig = plt.figure(figsize=(6, 4))
            plt.barh(dfp["feature"], dfp["score"])
            plt.xlabel("Feature importance (fallback)")
            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return np.array(Image.open(buf).convert("RGB")).astype(np.uint8)
    except Exception as e:
        print("FI fallback error:", e)
    return None

# ---------------- Heatmap ----------------
def _detect_locid_key(gj: dict) -> str:
    props = gj["features"][0]["properties"].keys()
    candidates = [
        "locationid", "location_id", "locid", "loc_id",
        "LocationID", "Location_Id", "Location_ID",
        "OBJECTID", "objectid", "id"
    ]
    for c in candidates:
        for k in props:
            if k.lower() == c.lower():
                return k
    return list(props)[0]

def heatmap_html():
    if zones_gj is None:
        return "<p>Upload <code>data/taxi_zones.geojson</code> to enable the map.</p>"

    id_key = _detect_locid_key(zones_gj)

    df = agg.copy()
    df["hour"] = df["pickup_hour"].dt.hour.astype(int)
    df["pressure"] = df["gap"] / (df["supply"] + 1e-9)
    kpis = df.groupby("pulocationid")["pressure"].mean().reset_index()

    m = folium.Map(location=[40.73, -73.95], tiles="cartodbpositron", zoom_start=11)
    folium.Choropleth(
        geo_data=zones_gj,
        data=kpis,
        columns=["pulocationid", "pressure"],
        key_on=f"feature.properties.{id_key}",
        fill_color="YlOrRd",
        fill_opacity=0.85,
        line_opacity=0.2,
        legend_name="Average pressure (dataset)"
    ).add_to(m)

    # Optional tooltip (Zone/Borough if present)
    try:
        from folium.features import GeoJson, GeoJsonTooltip
        props0 = zones_gj["features"][0]["properties"]
        fields, aliases = [], []
        for cand, label in [("zone","Zone"), ("borough","Borough"), (id_key,"LocationID")]:
            if cand in props0:
                fields.append(cand); aliases.append(label)
        if fields:
            GeoJson(
                zones_gj,
                name="zones",
                tooltip=GeoJsonTooltip(fields=fields, aliases=aliases, sticky=False)
            ).add_to(m)
    except Exception:
        pass

    return m._repr_html_()

# ---------------- Core recommend ----------------
def recommend(loc_id, hour, dow, month, use_latest, demand_manual, supply_manual):
    try:
        pulocationid = int(loc_id)
        zone_label = id_to_label.get(pulocationid, str(pulocationid))

        # Demand/supply: live vs manual
        if use_latest:
            zrows = agg[agg["pulocationid"] == pulocationid]
            if not zrows.empty:
                latest = zrows.sort_values("pickup_hour").iloc[-1]
                demand = int(latest["demand"])
                supply = int(latest["supply"])
                gap = int(latest["gap"])
                src = f"latest in data ({latest['pickup_hour']})"
            else:
                demand = supply = gap = 0
                src = "no history; using zeros"
        else:
            demand = int(demand_manual)
            supply = int(supply_manual)
            gap = demand - supply
            src = "manual entry"

        X = pd.DataFrame([{
            "hour": int(hour),
            "dow": int(dow),
            "month": int(month),
            "is_weekend": is_weekend(int(dow)),
            "pulocationid": pulocationid,
            "demand": demand,
            "supply": supply,
            "gap": gap
        }])[feature_cols]

        surge_pred = float(model.predict(X)[0])

        # Base fare (median by zone×hour if present, else zone median)
        bf_zone = basefare[basefare["pulocationid"] == pulocationid].copy()
        if not bf_zone.empty:
            bf_zone["hod"] = bf_zone["pickup_hour"].dt.hour.astype(int)
            pick = bf_zone[bf_zone["hod"] == int(hour)]
            base_fare = float((pick["base_fare"].median() if not pick.empty else bf_zone["base_fare"].median()))
        else:
            base_fare = 12.0

        rec_price = float(np.clip(base_fare * surge_pred, 5.0, 150.0))

        # Bonus: combine learned hourly suggestion and real-time pressure
        x_now = float(pressure_x(gap, supply, int(hour)))
        bonus = float(np.clip(2.0 + 3.0*(1.0/(1.0+np.exp(-x_now))), 0.0, 12.0))

        # Acceptance (illustrative)
        p_base = accept_prob(base_fare, base_fare)
        p_ml = accept_prob(rec_price, base_fare)

        # ETA
        eta_text = "N/A"
        if eta is not None:
            z = eta[eta["pulocationid"] == pulocationid].copy()
            if not z.empty:
                z["hod"] = z["pickup_hour"].dt.hour
                match = z[z["hod"] == int(hour)]
                val = float((match["median_eta_min"].median() if not match.empty else z["median_eta_min"].median()))
                eta_text = f"{val:.1f} min"

        # SHAP image
        shap_img = shap_bar_np(X)

        # Logging
        log_event({
            "ts": datetime.utcnow().isoformat(),
            "zone_label": zone_label,
            "pulocationid": pulocationid,
            "hour": int(hour), "dow": int(dow), "month": int(month),
            "demand": int(demand), "supply": int(supply), "gap": int(gap),
            "surge_pred": float(surge_pred), "base_fare": float(base_fare),
            "recommended_price": float(rec_price), "driver_bonus": float(bonus),
            "p_acc_base": float(p_base), "p_acc_ml": float(p_ml),
        })

        one_liner = (
            f"Zone: {zone_label} | Hour: {hour} | DOW: {dow} | Month: {month} | "
            f"Src: {src} | Demand={demand} Supply={supply} Gap={gap} → "
            f"Surge ≈ {surge_pred:.2f}; Base ${base_fare:.2f} ⇒ Price ${rec_price:.2f}; "
            f"Bonus ${bonus:.2f}; Accept {p_base:.2f}→{p_ml:.2f}; ETA {eta_text}"
        )

        explain_md = """
### What’s happening
- **Model:** XGBoost predicts surge from hour/dow/month/weekend/zone/demand/supply/gap.  
- **Price:** `base_fare × surge_pred` (clipped).  
- **Bonus:** Real-time pressure nudges bonus (illustrative).  
- **Acceptance:** Simple curve to show the trade-off (illustrative).  
- **Logging:** Saved to `data/events.db`.  
"""

        exec_md = (
            "### Executive summary\n"
            f"- **Context:** {zone_label}, Hour {hour}, DOW {dow}, Month {month} "
            f"(demand {demand}, supply {supply}, gap {gap}; src: {src}).\n"
            f"- **Model:** Predicted surge ≈ {surge_pred:.2f}; base fare ${base_fare:.2f}; ETA {eta_text}.\n"
            f"- **Recommendation:** Price **${rec_price:.2f}**; Driver bonus **${bonus:.2f}**.\n"
            f"- **Impact:** Acceptance baseline {p_base:.2f} → ML {p_ml:.2f} (illustrative)."
        )

        out_df = pd.DataFrame([{
            "pulocationid": pulocationid, "hour": int(hour), "dow(0=Sun)": int(dow), "month": int(month),
            "demand": int(demand), "supply": int(supply), "gap": int(gap),
            "surge_pred": round(surge_pred, 3), "base_fare": round(base_fare, 2),
            "recommended_price": round(rec_price, 2), "driver_bonus": round(bonus, 2),
            "p_acc_base": round(p_base, 3), "p_acc_ml": round(p_ml, 3), "eta_median_hist": eta_text
        }])

        return out_df, one_liner, explain_md, exec_md, shap_img

    except Exception as e:
        err = f"**Error:** {type(e).__name__}: {e}"
        print(err)
        empty = pd.DataFrame([{
            "pulocationid": "", "hour": "", "dow(0=Sun)": "", "month": "",
            "demand": "", "supply": "", "gap": "",
            "surge_pred": "", "base_fare": "", "recommended_price": "",
            "driver_bonus": "", "p_acc_base": "", "p_acc_ml": "", "eta_median_hist": ""
        }])
        return empty, err, "", err, None

# ---------------- UI ----------------
def now_defaults():
    now = datetime.utcnow()
    py_dow = now.weekday()    # Mon=0..Sun=6
    dow = (py_dow + 1) % 7    # Sun=0..Sat=6
    return now.hour, dow, now.month

h0, d0, m0 = now_defaults()

with gr.Blocks(title="NYC Pricing & Incentive Recommender") as demo:
    gr.Markdown("# NYC Pricing & Incentive Recommender")

    with gr.Tab("Recommend"):
        with gr.Row():
            loc = gr.Dropdown(choices=[(lbl, i) for i, lbl in zone_choices], value=zone_choices[0][0], label="Pickup zone")
            hour = gr.Slider(0, 23, value=h0, step=1, label="Hour of day")
            dow = gr.Slider(0, 6, value=d0, step=1, label="Day of week (0=Sun)")
            month = gr.Slider(1, 12, value=m0, step=1, label="Month")

        use_latest = gr.Checkbox(value=True, label="Use latest demand/supply from data for this zone")
        with gr.Row():
            demand_in = gr.Number(value=20, precision=0, label="Demand (manual if unchecked)")
            supply_in = gr.Number(value=15, precision=0, label="Supply (manual if unchecked)")

        run_btn = gr.Button("Recommend")

        out_df = gr.Dataframe(label="Recommendation", wrap=True)
        one_line = gr.Markdown()
        explain_md = gr.Markdown()
        exec_md = gr.Markdown()
        shap_img = gr.Image(type="numpy", label="Top feature contributions (SHAP)")

        run_btn.click(
            recommend,
            inputs=[loc, hour, dow, month, use_latest, demand_in, supply_in],
            outputs=[out_df, one_line, explain_md, exec_md, shap_img]
        )

    with gr.Tab("NYC heatmap"):
        gr.Markdown("Average pressure (dataset) by pickup zone.")
        gr.HTML(heatmap_html())

    with gr.Tab("Logs"):
        gr.Markdown("Recent events are saved to `data/events.db`. Export a CSV snapshot:")
        dl_btn = gr.Button("Export last 5000 events as CSV")
        dl_file = gr.File(label="Download events_export.csv")
        dl_btn.click(lambda: export_events(), outputs=[dl_file])

if __name__ == "__main__":
    demo.launch()
