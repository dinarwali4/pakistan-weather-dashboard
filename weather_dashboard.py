import streamlit as st
import matplotlib.pyplot as plt
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
st.title("Weather Forecast Dashboard for Pakistan")
st.caption("Pakistan (24-37N, 61-77E) | Live GFS data from NOAA")

# --- Sidebar Controls ---
st.sidebar.header("Forecast Settings")

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


# --- Helper ---
def _select_lead_time(plot_data, lt_hr):
    if "lead_time" in plot_data.dims:
        target_lt = np.timedelta64(lt_hr, "h")
        if target_lt in plot_data.lead_time.values:
            return plot_data.sel(lead_time=target_lt)
        return plot_data.isel(lead_time=0)
    return plot_data


# --- Render Weather Panel ---
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

    # --- Region Averages ---
    st.subheader(f"Region Averages — {label}")
    cols = st.columns(len(display_vars))
    for col, var in zip(cols, display_vars):
        info = VARIABLE_INFO.get(var, {"name": var, "unit": ""})
        data = _select_lead_time(ds_display[var].squeeze(), lead_time_hr).squeeze()
        val = float(data.mean(skipna=True).values)
        col.metric(info["name"], f"{val:.1f} {info['unit']}")

    # --- Weather Maps ---
    st.subheader(f"Weather Maps — {label}")
    for row_start in range(0, len(display_vars), 3):
        row_vars = display_vars[row_start: row_start + 3]
        fig, axes = plt.subplots(1, len(row_vars), figsize=(7 * len(row_vars), 5))
        if len(row_vars) == 1:
            axes = [axes]
        for ax, var in zip(axes, row_vars):
            info = VARIABLE_INFO.get(var, {"name": var, "unit": "", "cmap": "viridis"})
            plot_data = _select_lead_time(ds_display[var].squeeze(), lead_time_hr).squeeze()
            plot_data.plot(ax=ax, cmap=info["cmap"], add_colorbar=True)
            ax.set_title(f"{info['name']} ({info['unit']})")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # --- City Forecasts ---
    st.subheader(f"City Forecasts — {label}")
    city_rows = get_city_forecast(ds_display, lead_time_hr)
    if city_rows:
        df = pd.DataFrame(city_rows).set_index("City")
        st.dataframe(df, use_container_width=True)

    # --- Time Series ---
    st.subheader(f"City Time Series — {label}")
    if st.button("Load 10-Day Time Series", key=f"ts_btn_{label}"):
        with st.spinner("Fetching 10-day forecast (every 12h)..."):
            ts_ds = fetch_timeseries_data()
        if ts_ds is not None:
            ts_ds = compute_wind_speed(ts_ds)
            ts_display = convert_units(ts_ds)

            ts_col1, ts_col2 = st.columns(2)
            with ts_col1:
                selected_city = st.selectbox("City", list(PAKISTAN_CITIES.keys()), key=f"tsc_{label}")
            with ts_col2:
                ts_opts = {v: VARIABLE_INFO.get(v, {"name": v})["name"] for v in display_vars}
                selected_var = st.selectbox("Variable", list(ts_opts.keys()),
                                            format_func=lambda x: ts_opts[x], key=f"tsv_{label}")

            hours, values = get_city_timeseries(ts_display, selected_city, selected_var)
            if hours and values:
                info = VARIABLE_INFO.get(selected_var, {"name": selected_var, "unit": ""})
                fig_ts, ax_ts = plt.subplots(figsize=(12, 4))
                ax_ts.plot(hours, values, marker="o", markersize=4, linewidth=1.5, color="#1f77b4")
                ax_ts.axvline(x=lead_time_hr, color="red", linestyle="--", alpha=0.7,
                              label=f"Selected: +{lead_time_hr}h")
                ax_ts.set_xlabel("Lead Time (hours)")
                ax_ts.set_ylabel(f"{info['name']} ({info['unit']})")
                ax_ts.set_title(f"{info['name']} — 10 Day Forecast for {selected_city}")
                ax_ts.legend()
                ax_ts.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_ts)
                plt.close(fig_ts)
    else:
        st.caption("Click the button above to load the full 10-day time series.")


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

    with st.expander("View Raw Data"):
        st.write(ds)

    st.markdown("---")
    st.caption("Data: NOAA GFS via direct HTTP | Region: Pakistan (24-37N, 61-77E)")
elif data_source != "Upload NetCDF":
    st.warning("No data loaded. Check your settings and internet connection.")
