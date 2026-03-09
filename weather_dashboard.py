import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Pakistan Weather Dashboard")
st.title("🌦️ Weather Forecast Dashboard for Pakistan")
st.caption("Pakistan (24-37N, 61-77E) | Live GFS data from NOAA")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Forecast Settings")

data_source = st.sidebar.radio(
    "Data Source",
    ["GFS Forecast", "Upload NetCDF"],
)

if data_source != "Upload NetCDF":
    init_date = st.sidebar.date_input(
        "Forecast Initialization Date",
        value=date.today() - timedelta(days=1),
        min_value=date(2025, 1, 1),
        max_value=date.today(),
    )
    init_time = f"{init_date.isoformat()}T00:00:00"

    lead_time_hr = st.sidebar.slider(
        "Lead Time (hours ahead)",
        min_value=6,
        max_value=240,
        value=24,
        step=6,
    )
else:
    lead_time_hr = 6

# --- City Selection (Sidebar) ---
st.sidebar.header("🏙️ City Selection")
all_cities = list(PAKISTAN_CITIES.keys())
selected_city = st.sidebar.selectbox("Select City", all_cities, index=0)
compare_cities = st.sidebar.multiselect(
    "Compare Cities",
    all_cities,
    default=["Islamabad", "Lahore", "Karachi"],
)


# --- Data Fetching ---
@st.cache_data(show_spinner=False)
def load_uploaded_file(file_bytes, filename):
    tmp_path = os.path.join("/tmp", filename)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    return xr.open_dataset(tmp_path)


def fetch_data():
    if data_source == "GFS Forecast":
        with st.spinner(f"Fetching GFS forecast (+{lead_time_hr}h) from NOAA..."):
            return fetch_gfs_forecast(init_time, lead_hours=[lead_time_hr])
    elif data_source == "Upload NetCDF":
        uploaded = st.sidebar.file_uploader("Upload forecast NetCDF", type=["nc"])
        if uploaded is not None:
            return load_uploaded_file(uploaded.getvalue(), uploaded.name)
        else:
            st.info("Upload a forecast NetCDF file to visualize it.")
            return None
    return None


def fetch_timeseries_data():
    hours = list(range(6, 241, 12))
    return fetch_gfs_forecast(init_time, lead_hours=hours)


# --- Helpers ---
def _select_lead_time(plot_data, lt_hr):
    if "lead_time" in plot_data.dims:
        target_lt = np.timedelta64(lt_hr, "h")
        if target_lt in plot_data.lead_time.values:
            return plot_data.sel(lead_time=target_lt)
        return plot_data.isel(lead_time=0)
    return plot_data


def _get_city_value(ds_display, city_name, var_key, lt_hr):
    """Extract a single variable value for a city."""
    if city_name not in PAKISTAN_CITIES:
        return None
    lat, lon = PAKISTAN_CITIES[city_name]
    try:
        val = ds_display[var_key].sel(lat=lat, lon=lon, method="nearest")
        if "lead_time" in val.dims:
            target_lt = np.timedelta64(lt_hr, "h")
            if target_lt in val.lead_time.values:
                val = val.sel(lead_time=target_lt)
            else:
                val = val.isel(lead_time=0)
        return round(float(val.squeeze().values), 1)
    except Exception:
        return None


# --- Weather condition description ---
def _weather_condition(temp, humidity, precip, wind_speed):
    """Generate a simple weather condition string from values."""
    parts = []
    if temp is not None:
        if temp > 40:
            parts.append("🔥 Extreme Heat")
        elif temp > 35:
            parts.append("☀️ Very Hot")
        elif temp > 25:
            parts.append("🌤️ Warm")
        elif temp > 15:
            parts.append("⛅ Mild")
        elif temp > 5:
            parts.append("🌥️ Cool")
        elif temp > 0:
            parts.append("❄️ Cold")
        else:
            parts.append("🥶 Freezing")

    if precip is not None and precip > 0.5:
        if precip > 50:
            parts.append("🌧️ Heavy Rain")
        elif precip > 10:
            parts.append("🌦️ Rain")
        else:
            parts.append("🌂 Light Rain")

    if wind_speed is not None and wind_speed > 10:
        parts.append("💨 Windy")

    if humidity is not None:
        if humidity > 80:
            parts.append("💧 Humid")
        elif humidity < 20:
            parts.append("🏜️ Dry")

    return " | ".join(parts) if parts else "🌤️ Fair"


# --- Render City Detail Card ---
def render_city_detail(ds_display, city_name, lt_hr):
    """Render a detailed weather card for a single city."""
    lat, lon = PAKISTAN_CITIES[city_name]

    display_vars = ["t2m", "tp", "wind_speed", "r2m", "msl"]
    available_vars = [v for v in display_vars if v in ds_display]

    # Gather all values
    values = {}
    for var in available_vars:
        values[var] = _get_city_value(ds_display, city_name, var, lt_hr)

    # Weather condition
    condition = _weather_condition(
        values.get("t2m"), values.get("r2m"),
        values.get("tp"), values.get("wind_speed"),
    )

    st.markdown(f"### 📍 {city_name}")
    st.markdown(f"**Coordinates:** {lat:.2f}°N, {lon:.2f}°E &nbsp;|&nbsp; **Forecast:** +{lt_hr}h")
    st.markdown(f"**Conditions:** {condition}")

    # Metric cards in columns
    cols = st.columns(len(available_vars))
    for col, var in zip(cols, available_vars):
        info = VARIABLE_INFO.get(var, {"name": var, "unit": ""})
        val = values.get(var)
        if val is not None:
            col.metric(info["name"], f"{val:.1f} {info['unit']}")
        else:
            col.metric(info["name"], "N/A")


# --- Render City Comparison ---
def render_city_comparison(ds_display, cities, lt_hr):
    """Render a bar chart comparing multiple cities across all variables."""
    display_vars = ["t2m", "tp", "wind_speed", "r2m", "msl"]
    available_vars = [v for v in display_vars if v in ds_display]

    if not cities:
        st.info("Select cities to compare in the sidebar.")
        return

    # Build comparison data
    comp_data = {}
    for city in cities:
        comp_data[city] = {}
        for var in available_vars:
            comp_data[city][var] = _get_city_value(ds_display, city, var, lt_hr)

    # One chart per variable
    n_vars = len(available_vars)
    cols_per_row = min(3, n_vars)
    for row_start in range(0, n_vars, cols_per_row):
        row_vars = available_vars[row_start:row_start + cols_per_row]
        fig, axes = plt.subplots(1, len(row_vars), figsize=(6 * len(row_vars), 4))
        if len(row_vars) == 1:
            axes = [axes]
        for ax, var in zip(axes, row_vars):
            info = VARIABLE_INFO.get(var, {"name": var, "unit": "", "cmap": "viridis"})
            city_names = []
            vals = []
            for city in cities:
                v = comp_data[city].get(var)
                if v is not None:
                    city_names.append(city)
                    vals.append(v)
            if vals:
                colors = plt.cm.get_cmap("Set2")(np.linspace(0, 1, len(vals)))
                bars = ax.bar(city_names, vals, color=colors, edgecolor="white", linewidth=0.5)
                ax.set_title(f"{info['name']} ({info['unit']})", fontsize=12, fontweight="bold")
                ax.set_ylabel(info["unit"])
                ax.tick_params(axis="x", rotation=45)
                # Add value labels on bars
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f"{val:.1f}", ha="center", va="bottom", fontsize=9)
                ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# --- Render Weather Maps with City Markers ---
def render_weather_maps(ds_display, lt_hr):
    """Render weather maps with city locations marked."""
    preferred_order = ["t2m", "tp", "wind_speed", "r2m", "msl"]
    display_vars = [v for v in preferred_order if v in ds_display]

    for row_start in range(0, len(display_vars), 3):
        row_vars = display_vars[row_start: row_start + 3]
        fig, axes = plt.subplots(1, len(row_vars), figsize=(7 * len(row_vars), 5.5))
        if len(row_vars) == 1:
            axes = [axes]
        for ax, var in zip(axes, row_vars):
            info = VARIABLE_INFO.get(var, {"name": var, "unit": "", "cmap": "viridis"})
            plot_data = _select_lead_time(ds_display[var].squeeze(), lt_hr).squeeze()
            plot_data.plot(ax=ax, cmap=info["cmap"], add_colorbar=True)

            # Plot city markers
            for city, (clat, clon) in PAKISTAN_CITIES.items():
                ax.plot(clon, clat, "k^", markersize=6, markeredgecolor="white",
                        markeredgewidth=0.8)
                ax.annotate(city, (clon, clat), fontsize=6, fontweight="bold",
                            color="black", ha="left", va="bottom",
                            xytext=(2, 2), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      ec="gray", alpha=0.7, lw=0.5))

            # Highlight selected city
            sel_lat, sel_lon = PAKISTAN_CITIES[selected_city]
            ax.plot(sel_lon, sel_lat, "r*", markersize=14, markeredgecolor="white",
                    markeredgewidth=1)

            ax.set_title(f"{info['name']} ({info['unit']})", fontsize=11, fontweight="bold")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# --- Render All City Forecasts Table ---
def render_city_table(ds_display, lt_hr):
    """Render a styled dataframe with all city forecasts."""
    city_rows = get_city_forecast(ds_display, lt_hr)
    if city_rows:
        df = pd.DataFrame(city_rows).set_index("City")
        # Highlight the selected city
        def highlight_selected(row):
            if row.name == selected_city:
                return ["background-color: #fff3cd"] * len(row)
            return [""] * len(row)
        styled = df.style.apply(highlight_selected, axis=1).format("{:.1f}", na_rep="—")
        st.dataframe(styled, use_container_width=True)
    else:
        st.warning("No city data available.")


# --- Render Time Series ---
def render_timeseries(ds_display, display_vars, label):
    """Render time series section for a city."""
    if st.button("📈 Load 10-Day Time Series", key=f"ts_btn_{label}"):
        with st.spinner("Fetching 10-day forecast (every 12h)..."):
            ts_ds = fetch_timeseries_data()
        if ts_ds is not None:
            ts_ds = compute_wind_speed(ts_ds)
            ts_display = convert_units(ts_ds)

            ts_col1, ts_col2 = st.columns(2)
            with ts_col1:
                ts_city = st.selectbox("City", all_cities,
                                       index=all_cities.index(selected_city),
                                       key=f"tsc_{label}")
            with ts_col2:
                ts_opts = {v: VARIABLE_INFO.get(v, {"name": v})["name"] for v in display_vars}
                ts_var = st.selectbox("Variable", list(ts_opts.keys()),
                                      format_func=lambda x: ts_opts[x], key=f"tsv_{label}")

            hours, values = get_city_timeseries(ts_display, ts_city, ts_var)
            if hours and values:
                info = VARIABLE_INFO.get(ts_var, {"name": ts_var, "unit": ""})
                fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
                ax_ts.plot(hours, values, marker="o", markersize=5, linewidth=2,
                           color="#1f77b4", label=ts_city)
                ax_ts.fill_between(hours, values, alpha=0.1, color="#1f77b4")
                ax_ts.axvline(x=lead_time_hr, color="red", linestyle="--", alpha=0.7,
                              label=f"Selected: +{lead_time_hr}h")
                ax_ts.set_xlabel("Lead Time (hours)", fontsize=11)
                ax_ts.set_ylabel(f"{info['name']} ({info['unit']})", fontsize=11)
                ax_ts.set_title(f"{info['name']} — 10 Day Forecast for {ts_city}",
                                fontsize=13, fontweight="bold")
                ax_ts.legend(fontsize=10)
                ax_ts.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_ts)
                plt.close(fig_ts)
            else:
                st.warning(f"No time series data for {ts_city}.")
    else:
        st.caption("Click the button above to load the full 10-day time series.")


# --- Main Render Function ---
def render_weather_panel(ds, label):
    ds = compute_wind_speed(ds)
    ds_display = convert_units(ds)

    preferred_order = ["t2m", "tp", "wind_speed", "r2m", "msl"]
    display_vars = [v for v in preferred_order if v in ds_display]

    # --- Alerts ---
    alerts = check_alerts(ds_display, lead_time_hr)
    if alerts:
        for level, title, detail in alerts:
            if level == "error":
                st.error(f"**{title}** — {detail}")
            else:
                st.warning(f"**{title}** — {detail}")

    # ===========================================================
    # SECTION 1: Selected City Detail Card
    # ===========================================================
    st.markdown("---")
    render_city_detail(ds_display, selected_city, lead_time_hr)

    # ===========================================================
    # SECTION 2: City Comparison
    # ===========================================================
    st.markdown("---")
    st.subheader("📊 City Comparison")
    render_city_comparison(ds_display, compare_cities, lead_time_hr)

    # ===========================================================
    # SECTION 3: All Cities Table
    # ===========================================================
    st.markdown("---")
    st.subheader(f"🏙️ All City Forecasts — {label}")
    render_city_table(ds_display, lead_time_hr)

    # ===========================================================
    # SECTION 4: Weather Maps (with city markers)
    # ===========================================================
    st.markdown("---")
    st.subheader(f"🗺️ Weather Maps — {label}")
    render_weather_maps(ds_display, lead_time_hr)

    # ===========================================================
    # SECTION 5: Region Averages
    # ===========================================================
    st.markdown("---")
    st.subheader(f"📏 Region Averages — {label}")
    cols = st.columns(len(display_vars))
    for col, var in zip(cols, display_vars):
        info = VARIABLE_INFO.get(var, {"name": var, "unit": ""})
        data = _select_lead_time(ds_display[var].squeeze(), lead_time_hr).squeeze()
        val = float(data.mean(skipna=True).values)
        col.metric(info["name"], f"{val:.1f} {info['unit']}")

    # ===========================================================
    # SECTION 6: Time Series
    # ===========================================================
    st.markdown("---")
    st.subheader(f"📈 City Time Series — {label}")
    render_timeseries(ds_display, display_vars, label)


# --- Main App ---
ds = fetch_data()

if ds is not None:
    mar_tab, apr_tab, may_tab = st.tabs(["March 2026", "April 2026", "May 2026"])

    with mar_tab:
        render_weather_panel(ds, "March 2026")
    with apr_tab:
        render_weather_panel(ds, "April 2026")
    with may_tab:
        render_weather_panel(ds, "May 2026")

    with st.expander("🔍 View Raw Data"):
        st.write(ds)

    st.markdown("---")
    st.caption("Data: NOAA GFS via direct HTTP | Region: Pakistan (24-37N, 61-77E)")
elif data_source != "Upload NetCDF":
    st.warning("No data loaded. Check your settings and internet connection.")
