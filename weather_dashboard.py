import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import pandas as pd
import os
from datetime import datetime, date, timedelta

from earth2studio_fetch import (
    fetch_gfs_forecast,
    convert_units,
    compute_wind_speed,
    get_city_forecast,
    get_city_timeseries,
    check_alerts,
    VARIABLE_INFO,
    PAKISTAN_CITIES,
)

# =====================================================================
# Page Config
# =====================================================================
st.set_page_config(
    layout="wide",
    page_title="RegenX.eco Geospatial Climate Model",
    page_icon="🇵🇰",
)

# =====================================================================
# Custom CSS
# =====================================================================
st.markdown("""
<style>
/* -------- Global -------- */
.block-container { padding: 1.2rem 2rem 2rem 2rem; max-width: 1200px; }
h1, h2, h3 { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

/* -------- Header -------- */
.hdr {
    background: linear-gradient(135deg, #0a2540 0%, #0969da 70%, #54aeff 100%);
    border-radius: 14px; padding: 1.8rem 2.2rem; margin-bottom: 1.4rem;
    color: #fff;
}
.hdr h1 { margin:0; font-size:1.7rem; font-weight:700; color:#fff; }
.hdr p  { margin:0.3rem 0 0; font-size:0.85rem; opacity:0.85; }

/* -------- Metric tile -------- */
.mt {
    background:#fff; border:1px solid #d0d7de; border-radius:10px;
    padding:0.85rem 0.6rem; text-align:center; height:110px;
    display:flex; flex-direction:column; justify-content:center;
}
.mt .ic  { font-size:1.3rem; margin-bottom:0.1rem; }
.mt .lb  { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.5px; color:#656d76; font-weight:600; margin-bottom:0.2rem; }
.mt .vl  { font-size:1.45rem; font-weight:700; color:#1f2328; line-height:1.15; }
.mt .un  { font-size:0.75rem; font-weight:400; color:#656d76; }

/* -------- City hero -------- */
.ch {
    background: linear-gradient(135deg, #f0f7ff 0%, #dbeafe 100%);
    border:1px solid #b6d4fe; border-radius:14px;
    padding:1.4rem 1.8rem; margin-bottom:0.6rem;
}
.ch .nm  { font-size:1.45rem; font-weight:700; color:#0a2540; margin:0 0 0.1rem; }
.ch .mt2 { font-size:0.8rem; color:#4b6a8a; margin:0 0 0.55rem; }
.ch .cd  {
    display:inline-block; background:#fff; border:1px solid #b6d4fe;
    border-radius:8px; padding:0.3rem 0.8rem; font-size:0.9rem; color:#1f2328;
}

/* -------- Section header -------- */
.sh {
    font-size:1.02rem; font-weight:700; color:#1f2328;
    margin:1.5rem 0 0.6rem; padding-bottom:0.4rem;
    border-bottom:2px solid #0969da;
}
.sh .bg {
    background:#0969da; color:#fff; font-size:0.6rem;
    padding:0.12rem 0.5rem; border-radius:10px; font-weight:600;
    text-transform:uppercase; letter-spacing:0.4px;
    margin-left:0.5rem; vertical-align:middle;
}

/* -------- Comparison city card -------- */
.cc {
    background:#fff; border:1px solid #d0d7de; border-radius:10px;
    padding:0.9rem 1rem; height:100%;
}
.cc .cn {
    font-size:0.85rem; font-weight:700; color:#0a2540;
    margin-bottom:0.5rem; padding-bottom:0.35rem;
    border-bottom:2px solid #54aeff; text-align:center;
}
.cc .cr {
    display:flex; justify-content:space-between; align-items:center;
    padding:0.22rem 0; font-size:0.78rem; color:#1f2328;
}
.cc .cr .cl { color:#656d76; }
.cc .cr .cv { font-weight:600; }

/* -------- Region avg tile -------- */
.ra {
    background:#fff; border:1px solid #d0d7de; border-radius:10px;
    padding:0.85rem 0.6rem; text-align:center; height:120px;
    display:flex; flex-direction:column; justify-content:center;
}
.ra .ic  { font-size:1.2rem; margin-bottom:0.1rem; }
.ra .lb  { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.5px; color:#656d76; font-weight:600; margin-bottom:0.15rem; }
.ra .vl  { font-size:1.35rem; font-weight:700; color:#1f2328; line-height:1.15; }
.ra .rng { font-size:0.65rem; color:#656d76; margin-top:0.2rem; }

/* -------- Table wrap -------- */
.tw { border:1px solid #d0d7de; border-radius:12px; overflow:hidden; margin-top:0.3rem; }

/* -------- Footer -------- */
.ft { text-align:center; color:#8b949e; font-size:0.75rem; padding:1rem 0 0.3rem; border-top:1px solid #d8dee4; margin-top:1.8rem; }

/* -------- Sidebar -------- */
section[data-testid="stSidebar"] { background:#f6f8fa; }
section[data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
.sb {
    text-align:center; padding:0.4rem 0 0.7rem; margin-bottom:0.8rem;
    border-bottom:1px solid #d8dee4;
}
.sb .t { font-size:0.95rem; font-weight:700; color:#0a2540; margin:0.3rem 0 0; }
.sb .s { font-size:0.68rem; color:#656d76; margin:0; }

/* -------- Tabs -------- */
button[data-baseweb="tab"] { font-weight:600 !important; }

/* -------- Hide chrome -------- */
#MainMenu, header, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# Header
# =====================================================================
st.markdown("""
<div class="hdr">
    <h1>🌍 RegenX.eco Geospatial Climate Model</h1>
    <p>Seasonal Weather Forecast &bull; 10 major cities &bull; Live GFS data from NOAA</p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# Sidebar
# =====================================================================
st.sidebar.markdown("""
<div class="sb">
    <div style="font-size:1.8rem;">🌍</div>
    <div class="t">RegenX.eco</div>
    <div class="s">Geospatial Climate Model</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("##### Data Source")
data_source = st.sidebar.radio("src", ["GFS Forecast", "Upload NetCDF"], label_visibility="collapsed")

if data_source != "Upload NetCDF":
    st.sidebar.markdown("##### Forecast Date")
    init_date = st.sidebar.date_input(
        "dt", value=date.today() - timedelta(days=1),
        min_value=date(2025, 1, 1), max_value=date.today(),
        label_visibility="collapsed",
    )
    init_time = f"{init_date.isoformat()}T00:00:00"

    st.sidebar.markdown("##### Lead Time (hours)")
    lead_time_hr = st.sidebar.slider("lt", 6, 240, 24, step=6, label_visibility="collapsed")
    valid_time = datetime.fromisoformat(init_time) + timedelta(hours=lead_time_hr)
    st.sidebar.caption(f"Valid: **{valid_time.strftime('%b %d, %Y %H:%M UTC')}**")
else:
    lead_time_hr = 6

st.sidebar.markdown("---")
st.sidebar.markdown("##### Primary City")
all_cities = list(PAKISTAN_CITIES.keys())
selected_city = st.sidebar.selectbox("pc", all_cities, index=0, label_visibility="collapsed")
st.sidebar.markdown("##### Compare Cities")
compare_cities = st.sidebar.multiselect(
    "cc", all_cities, default=["Islamabad", "Lahore", "Karachi"],
    label_visibility="collapsed",
)

# =====================================================================
# Data Loading
# =====================================================================
@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes, filename):
    tmp = os.path.join("/tmp", filename)
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    return xr.open_dataset(tmp)


def fetch_data():
    if data_source == "GFS Forecast":
        with st.spinner(f"Fetching GFS forecast (+{lead_time_hr}h)..."):
            return fetch_gfs_forecast(init_time, lead_hours=[lead_time_hr])
    elif data_source == "Upload NetCDF":
        uploaded = st.sidebar.file_uploader("Upload NetCDF", type=["nc"])
        if uploaded:
            return load_uploaded_file(uploaded.getvalue(), uploaded.name)
        st.info("Upload a forecast NetCDF file to begin.")
    return None


def fetch_timeseries_data():
    return fetch_gfs_forecast(init_time, lead_hours=list(range(6, 241, 12)))


# =====================================================================
# Helpers
# =====================================================================
ICONS = {"t2m": "🌡️", "tp": "🌧️", "wind_speed": "💨", "r2m": "💧", "msl": "📊"}
COLORS = {"t2m": "#ef4444", "tp": "#3b82f6", "wind_speed": "#f59e0b",
           "r2m": "#22c55e", "msl": "#8b5cf6"}
DVARS = ["t2m", "tp", "wind_speed", "r2m", "msl"]


def _sel(data, lt):
    if "lead_time" in data.dims:
        t = np.timedelta64(lt, "h")
        return data.sel(lead_time=t) if t in data.lead_time.values else data.isel(lead_time=0)
    return data


def _cv(ds_d, city, var, lt):
    if city not in PAKISTAN_CITIES:
        return None
    la, lo = PAKISTAN_CITIES[city]
    try:
        v = _sel(ds_d[var].sel(lat=la, lon=lo, method="nearest"), lt)
        return round(float(v.squeeze().values), 1)
    except Exception:
        return None


def _cond(t, h, p, w):
    parts = []
    if t is not None:
        if t > 40:   parts.append("🔥 Extreme Heat")
        elif t > 35: parts.append("☀️ Very Hot")
        elif t > 25: parts.append("🌤️ Warm")
        elif t > 15: parts.append("⛅ Mild")
        elif t > 5:  parts.append("🌥️ Cool")
        elif t > 0:  parts.append("❄️ Cold")
        else:        parts.append("🥶 Freezing")
    if p is not None:
        if p > 50:    parts.append("🌧️ Heavy Rain")
        elif p > 10:  parts.append("🌦️ Rain")
        elif p > 0.5: parts.append("🌂 Light Rain")
        else:         parts.append("☀️ No Rain")
    if w is not None and w > 10:
        parts.append("💨 Windy")
    if h is not None:
        if h > 80:   parts.append("💧 Humid")
        elif h < 20: parts.append("🏜️ Dry")
    return " &bull; ".join(parts) if parts else "🌤️ Fair"


# =====================================================================
# SECTION: City Hero Card
# =====================================================================
def ui_city_hero(ds_d, city, lt):
    la, lo = PAKISTAN_CITIES[city]
    avail = [v for v in DVARS if v in ds_d]
    vals = {v: _cv(ds_d, city, v, lt) for v in avail}
    cond = _cond(vals.get("t2m"), vals.get("r2m"), vals.get("tp"), vals.get("wind_speed"))

    st.markdown(f"""
    <div class="ch">
        <div class="nm">📍 {city}</div>
        <div class="mt2">{la:.2f}°N, {lo:.2f}°E &bull; +{lt}h forecast</div>
        <div class="cd">{cond}</div>
    </div>
    """, unsafe_allow_html=True)

    # Metric tiles via st.columns
    cols = st.columns(len(avail))
    for i, v in enumerate(avail):
        info = VARIABLE_INFO.get(v, {"name": v, "unit": ""})
        ic = ICONS.get(v, "")
        val = vals.get(v)
        vs = f"{val:.1f}" if val is not None else "—"
        with cols[i]:
            st.markdown(f"""
            <div class="mt">
                <div class="ic">{ic}</div>
                <div class="lb">{info['name']}</div>
                <div class="vl">{vs} <span class="un">{info['unit']}</span></div>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# SECTION: City Comparison (native st.columns — no raw HTML grid)
# =====================================================================
def ui_comparison(ds_d, cities, lt):
    avail = [v for v in DVARS if v in ds_d]
    if not cities:
        st.info("Select cities to compare in the sidebar.")
        return

    comp = {c: {v: _cv(ds_d, c, v, lt) for v in avail} for c in cities}

    # Render one card per city using st.columns
    n = len(cities)
    cols = st.columns(min(n, 4))
    for idx, city in enumerate(cities):
        with cols[idx % 4]:
            rows_html = ""
            for v in avail:
                info = VARIABLE_INFO.get(v, {"name": v, "unit": ""})
                ic = ICONS.get(v, "")
                val = comp[city].get(v)
                vs = f"{val:.1f} {info['unit']}" if val is not None else "—"
                rows_html += f'<div class="cr"><span class="cl">{ic} {info["name"]}</span><span class="cv">{vs}</span></div>'
            st.markdown(f"""
            <div class="cc">
                <div class="cn">{city}</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

    # If there are more than 4 cities, wrap to next row
    if n > 4:
        cols2 = st.columns(min(n - 4, 4))
        for idx, city in enumerate(cities[4:]):
            with cols2[idx % 4]:
                rows_html = ""
                for v in avail:
                    info = VARIABLE_INFO.get(v, {"name": v, "unit": ""})
                    ic = ICONS.get(v, "")
                    val = comp[city].get(v)
                    vs = f"{val:.1f} {info['unit']}" if val is not None else "—"
                    rows_html += f'<div class="cr"><span class="cl">{ic} {info["name"]}</span><span class="cv">{vs}</span></div>'
                st.markdown(f"""
                <div class="cc">
                    <div class="cn">{city}</div>
                    {rows_html}
                </div>
                """, unsafe_allow_html=True)

    # Grouped bar chart if 2+ cities
    if len(cities) >= 2:
        st.markdown("")
        _comparison_chart(comp, cities, avail)


def _comparison_chart(comp, cities, avail):
    n_c = len(cities)
    n_v = len(avail)
    bar_h = 0.14
    gap = 0.35
    palette = list(COLORS.values())

    fig, ax = plt.subplots(figsize=(10, max(3, n_c * 0.9 + 1)))
    y_base = np.arange(n_c) * (n_v * bar_h + gap)

    for j, v in enumerate(avail):
        info = VARIABLE_INFO.get(v, {"name": v, "unit": ""})
        raw = [comp[c].get(v, 0) or 0 for c in cities]
        mx = max(abs(x) for x in raw) if any(x != 0 for x in raw) else 1
        normed = [x / mx * 100 if mx else 0 for x in raw]
        y = y_base + j * bar_h
        color = palette[j % len(palette)]
        bars = ax.barh(y, normed, height=bar_h * 0.85, color=color,
                       alpha=0.82, edgecolor="white", linewidth=0.5,
                       label=info["name"])
        for i, (bar, rv) in enumerate(zip(bars, raw)):
            ax.text(max(bar.get_width() + 1, 3), y[i],
                    f" {rv:.1f} {info['unit']}", va="center", ha="left",
                    fontsize=8, color="#444", fontweight="500")

    ax.set_yticks(y_base + (n_v - 1) * bar_h / 2)
    ax.set_yticklabels(cities, fontsize=10, fontweight="600", color="#1f2328")
    ax.invert_yaxis()
    ax.set_xlim(0, 130)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.grid(axis="x", alpha=0.1, linestyle="-")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
              borderpad=0.6, handlelength=1.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================
# SECTION: All Cities Table
# =====================================================================
def ui_table(ds_d, lt):
    rows = get_city_forecast(ds_d, lt)
    if not rows:
        st.warning("No city data available.")
        return
    df = pd.DataFrame(rows).set_index("City")

    def hl(row):
        if row.name == selected_city:
            return ["background-color:#dbeafe; font-weight:700;"] * len(row)
        return [""] * len(row)

    styled = (
        df.style.apply(hl, axis=1).format("{:.1f}", na_rep="—")
        .set_properties(**{"text-align": "center", "font-size": "0.88rem"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#f6f8fa"), ("font-weight", "700"),
                ("font-size", "0.78rem"), ("text-align", "center"),
                ("color", "#1f2328"), ("padding", "8px 10px"),
            ]},
            {"selector": "td", "props": [("padding", "6px 10px")]},
        ])
    )
    st.dataframe(styled, use_container_width=True, height=420)


# =====================================================================
# SECTION: Weather Maps
# =====================================================================
def ui_maps(ds_d, lt):
    avail = [v for v in DVARS if v in ds_d]
    for row_start in range(0, len(avail), 3):
        row_vars = avail[row_start:row_start + 3]
        cols = st.columns(len(row_vars))
        for ci, var in enumerate(row_vars):
            with cols[ci]:
                info = VARIABLE_INFO.get(var, {"name": var, "unit": "", "cmap": "viridis"})
                data = _sel(ds_d[var].squeeze(), lt).squeeze()

                fig, ax = plt.subplots(figsize=(5.5, 4.8))
                data.plot(ax=ax, cmap=info["cmap"], add_colorbar=True,
                          cbar_kwargs={"shrink": 0.8, "pad": 0.03})

                for city, (cla, clo) in PAKISTAN_CITIES.items():
                    sel = city == selected_city
                    if sel:
                        ax.plot(clo, cla, "*", color="#e11d48", markersize=14,
                                markeredgecolor="white", markeredgewidth=1, zorder=10)
                    else:
                        ax.plot(clo, cla, "o", color="#0a2540", markersize=3.5,
                                markeredgecolor="white", markeredgewidth=0.5, zorder=9)
                    ax.annotate(
                        city, (clo, cla), fontsize=5 if not sel else 6.5,
                        fontweight="bold" if sel else "normal",
                        color="#be123c" if sel else "#1f2328",
                        ha="left", va="bottom", xytext=(3, 3),
                        textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white",
                                  ec="#be123c" if sel else "#ccc", alpha=0.85, lw=0.4),
                        zorder=11,
                    )
                ax.set_title(f"{info['name']} ({info['unit']})", fontsize=10, fontweight="bold", pad=6)
                ax.set_xlabel("Longitude", fontsize=8, color="#656d76")
                ax.set_ylabel("Latitude", fontsize=8, color="#656d76")
                ax.tick_params(labelsize=7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# =====================================================================
# SECTION: Region Averages
# =====================================================================
def ui_region(ds_d, lt):
    avail = [v for v in DVARS if v in ds_d]
    cols = st.columns(len(avail))
    for i, v in enumerate(avail):
        info = VARIABLE_INFO.get(v, {"name": v, "unit": ""})
        ic = ICONS.get(v, "")
        c = COLORS.get(v, "#0969da")
        data = _sel(ds_d[v].squeeze(), lt).squeeze()
        mn = float(data.mean(skipna=True).values)
        lo = float(data.min(skipna=True).values)
        hi = float(data.max(skipna=True).values)
        with cols[i]:
            st.markdown(f"""
            <div class="ra" style="border-top:3px solid {c};">
                <div class="ic">{ic}</div>
                <div class="lb">Avg {info['name']}</div>
                <div class="vl">{mn:.1f} <span class="un">{info['unit']}</span></div>
                <div class="rng">Min {lo:.1f} &bull; Max {hi:.1f}</div>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# SECTION: Time Series
# =====================================================================
def ui_timeseries(ds_d, label):
    avail = [v for v in DVARS if v in ds_d]

    if st.button("Load 10-Day Time Series", key=f"ts_{label}", type="primary",
                 use_container_width=True):
        with st.spinner("Fetching 10-day forecast (every 12h)..."):
            ts_ds = fetch_timeseries_data()
        if ts_ds is None:
            st.error("Could not load time series.")
            return
        ts_ds = compute_wind_speed(ts_ds)
        ts_d = convert_units(ts_ds)

        c1, c2 = st.columns(2)
        with c1:
            ts_city = st.selectbox("City", all_cities,
                                   index=all_cities.index(selected_city),
                                   key=f"tsc_{label}")
        with c2:
            opts = {v: VARIABLE_INFO.get(v, {"name": v})["name"] for v in avail}
            ts_var = st.selectbox("Variable", list(opts.keys()),
                                  format_func=lambda x: opts[x], key=f"tsv_{label}")

        hours, values = get_city_timeseries(ts_d, ts_city, ts_var)
        if hours and values:
            info = VARIABLE_INFO.get(ts_var, {"name": ts_var, "unit": ""})
            color = COLORS.get(ts_var, "#0969da")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(hours, values, marker="o", markersize=5, linewidth=2.2,
                    color=color, label=ts_city, zorder=5)
            ax.fill_between(hours, values, alpha=0.07, color=color)
            ax.axvline(x=lead_time_hr, color="#e11d48", linestyle="--",
                       linewidth=1.5, alpha=0.75, label=f"Current: +{lead_time_hr}h")
            ax.set_xlabel("Lead Time (hours)", fontsize=10, color="#656d76")
            ax.set_ylabel(f"{info['name']} ({info['unit']})", fontsize=10, color="#656d76")
            ax.set_title(f"{info['name']} — {ts_city}", fontsize=13, fontweight="bold", color="#1f2328")
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.12, linestyle="-")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning(f"No time series data for {ts_city}.")
    else:
        st.caption("Click above to load the full 10-day time series for any city.")


# =====================================================================
# Main Panel
# =====================================================================
def render_panel(ds, label):
    ds = compute_wind_speed(ds)
    ds_d = convert_units(ds)

    # Alerts
    alerts = check_alerts(ds_d, lead_time_hr)
    if alerts:
        for level, title, detail in alerts:
            if level == "error":
                st.error(f"**{title}** — {detail}")
            else:
                st.warning(f"**{title}** — {detail}")

    # 1 — City hero
    ui_city_hero(ds_d, selected_city, lead_time_hr)

    # 2 — City comparison
    st.markdown(f'<div class="sh">📊 City Comparison <span class="bg">Live</span></div>', unsafe_allow_html=True)
    ui_comparison(ds_d, compare_cities, lead_time_hr)

    # 3 — All cities table
    st.markdown(f'<div class="sh">🏙️ All City Forecasts <span class="bg">{label}</span></div>', unsafe_allow_html=True)
    ui_table(ds_d, lead_time_hr)

    # 4 — Weather maps
    st.markdown(f'<div class="sh">🗺️ Weather Maps <span class="bg">{label}</span></div>', unsafe_allow_html=True)
    ui_maps(ds_d, lead_time_hr)

    # 5 — Region averages
    st.markdown(f'<div class="sh">📏 Region Averages <span class="bg">{label}</span></div>', unsafe_allow_html=True)
    ui_region(ds_d, lead_time_hr)

    # 6 — Time series
    st.markdown(f'<div class="sh">📈 Time Series <span class="bg">{label}</span></div>', unsafe_allow_html=True)
    ui_timeseries(ds_d, label)


# =====================================================================
# App Entry
# =====================================================================
ds = fetch_data()

if ds is not None:
    t1, t2, t3 = st.tabs(["  March 2026  ", "  April 2026  ", "  May 2026  "])
    with t1:
        render_panel(ds, "March 2026")
    with t2:
        render_panel(ds, "April 2026")
    with t3:
        render_panel(ds, "May 2026")

    with st.expander("View Raw Dataset"):
        st.write(ds)

    st.markdown('<div class="ft">Created by <strong>RegenX.eco</strong> &bull; Data: NOAA GFS via direct HTTP &bull; Region: Pakistan (24-37°N, 61-77°E)</div>', unsafe_allow_html=True)
elif data_source != "Upload NetCDF":
    st.warning("No data loaded. Please check your settings and internet connection.")
