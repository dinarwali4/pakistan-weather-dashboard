"""
earth2studio_fetch.py
Fetches live weather data from NOAA GFS/GEFS via NVIDIA Earth2Studio.
Uses subprocess to avoid Streamlit async event loop conflicts.
No GPU required.
"""

import numpy as np
import xarray as xr
import streamlit as st
import os
import sys
import json
import hashlib
import subprocess
import tempfile
from datetime import datetime, timedelta

# Pakistan region bounds
PK_LAT_MIN = 24.0
PK_LAT_MAX = 37.0
PK_LON_MIN = 61.0
PK_LON_MAX = 77.0

# GFS has tp (precipitation), GEFS does not
GFS_VARIABLES = ["t2m", "tp", "u10m", "v10m", "r2m", "msl"]
GEFS_VARIABLES = ["t2m", "u10m", "v10m", "r2m", "msl", "d2m"]

VARIABLE_INFO = {
    "t2m": {"name": "2m Temperature", "unit": "°C", "cmap": "RdBu_r"},
    "tp": {"name": "Precipitation", "unit": "mm", "cmap": "Blues"},
    "wind_speed": {"name": "Wind Speed", "unit": "m/s", "cmap": "YlOrRd"},
    "r2m": {"name": "Relative Humidity", "unit": "%", "cmap": "YlGn"},
    "msl": {"name": "Sea Level Pressure", "unit": "hPa", "cmap": "coolwarm"},
    "d2m": {"name": "2m Dew Point", "unit": "°C", "cmap": "YlGn"},
}

PAKISTAN_CITIES = {
    "Islamabad": (33.69, 73.04),
    "Lahore": (31.55, 74.35),
    "Karachi": (24.86, 67.01),
    "Peshawar": (34.01, 71.58),
    "Quetta": (30.18, 67.00),
    "Faisalabad": (31.42, 73.08),
    "Multan": (30.20, 71.47),
    "Rawalpindi": (33.60, 73.05),
    "Hyderabad": (25.40, 68.37),
    "Gilgit": (35.92, 74.31),
}

ALERT_THRESHOLDS = {
    "t2m": {"high": 45.0, "high_label": "Extreme Heat (>45°C)", "low": 0.0, "low_label": "Frost/Freeze (<0°C)"},
    "tp": {"high": 50.0, "high_label": "Heavy Rain (>50mm)", "low": None, "low_label": None},
    "wind_speed": {"high": 17.0, "high_label": "High Wind (>60 km/h)", "low": None, "low_label": None},
    "r2m": {"high": 95.0, "high_label": "Very High Humidity (>95%)", "low": 15.0, "low_label": "Very Dry Air (<15%)"},
}

# --- Cache ---
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _get_cache_path(source_type, init_time, extra=""):
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{source_type}_{init_time}_{extra}"
    filename = hashlib.md5(key.encode()).hexdigest() + ".nc"
    return os.path.join(CACHE_DIR, filename)


# -----------------------------------------------------------
# Subprocess-based fetching (avoids Streamlit event loop clash)
# -----------------------------------------------------------

_FETCH_SCRIPT = '''
import sys, json, numpy as np, xarray as xr

args = json.loads(sys.argv[1])
source_type = args["source"]
init_time = args["init_time"]
lead_hours = args["lead_hours"]
variables = args["variables"]
output_path = args["output_path"]
lat_max, lat_min = {lat_max}, {lat_min}
lon_min, lon_max = {lon_min}, {lon_max}

time_arr = np.array([np.datetime64(init_time)])
lead_arr = np.array([np.timedelta64(h, "h") for h in lead_hours])
var_arr = np.array(variables)

if source_type == "gfs":
    from earth2studio.data import GFS_FX
    src = GFS_FX()
    da = src(time=time_arr, lead_time=lead_arr, variable=var_arr)
    da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    datasets = {{}}
    for v in variables:
        try:
            datasets[v] = da.sel(variable=v).drop_vars("variable")
        except Exception:
            pass
    ds = xr.Dataset(datasets)
    ds.to_netcdf(output_path)
    print("OK")

elif source_type == "gefs":
    from earth2studio.data import GEFS_FX
    n_members = args.get("n_members", 5)
    members = ["gec00"] + [f"gep{{i:02d}}" for i in range(1, n_members)]
    member_list = []
    for m in members:
        try:
            src = GEFS_FX(member=m)
            da = src(time=time_arr, lead_time=lead_arr, variable=var_arr)
            da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
            datasets = {{}}
            for v in variables:
                try:
                    datasets[v] = da.sel(variable=v).drop_vars("variable")
                except Exception:
                    pass
            member_list.append(xr.Dataset(datasets))
        except Exception as e:
            print(f"Skip {{m}}: {{e}}", file=sys.stderr)
    if member_list:
        combined = xr.concat(member_list, dim="member")
        combined["member"] = np.arange(len(member_list))
        combined.to_netcdf(output_path)
        print("OK")
    else:
        print("FAIL: no members fetched")
        sys.exit(1)
'''.format(lat_max=PK_LAT_MAX, lat_min=PK_LAT_MIN, lon_min=PK_LON_MIN, lon_max=PK_LON_MAX)


def _run_fetch_subprocess(source_type, init_time, lead_hours, variables, n_members=5):
    """Run Earth2Studio fetch in a subprocess to avoid event loop conflicts."""
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        output_path = tmp.name

    args_dict = {
        "source": source_type,
        "init_time": init_time,
        "lead_hours": lead_hours,
        "variables": variables,
        "output_path": output_path,
        "n_members": n_members,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as script_f:
        script_f.write(_FETCH_SCRIPT)
        script_path = script_f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path, json.dumps(args_dict)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            ds = xr.open_dataset(output_path)
            return ds
        else:
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            st.error(f"Fetch failed: {error_msg[-200:]}")
            return None
    except subprocess.TimeoutExpired:
        st.error("Data fetch timed out (>5 min). Try a more recent date.")
        return None
    except Exception as e:
        st.error(f"Fetch error: {e}")
        return None
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gfs_forecast(init_time, lead_hours=None, variables=None):
    if variables is None:
        variables = GFS_VARIABLES
    if lead_hours is None:
        lead_hours = [24]

    cache_path = _get_cache_path("gfs", init_time, f"h{'_'.join(map(str, lead_hours))}")
    if os.path.exists(cache_path):
        try:
            return xr.open_dataset(cache_path)
        except Exception:
            pass

    ds = _run_fetch_subprocess("gfs", init_time, lead_hours, variables)
    if ds is not None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            ds.to_netcdf(cache_path)
        except Exception:
            pass
    return ds


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gefs_ensemble(init_time, lead_hours=None, variables=None, n_members=5):
    if variables is None:
        variables = GEFS_VARIABLES
    if lead_hours is None:
        lead_hours = [24]

    cache_path = _get_cache_path("gefs", init_time, f"m{n_members}_h{'_'.join(map(str, lead_hours))}")
    if os.path.exists(cache_path):
        try:
            return xr.open_dataset(cache_path)
        except Exception:
            pass

    ds = _run_fetch_subprocess("gefs", init_time, lead_hours, variables, n_members=n_members)
    if ds is not None:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            ds.to_netcdf(cache_path)
        except Exception:
            pass
    return ds


# --- Processing functions (unchanged) ---

def convert_units(ds):
    result = ds.copy()
    if "t2m" in result:
        result["t2m"] = result["t2m"] - 273.15
    if "tp" in result:
        result["tp"] = result["tp"] * 1000
    if "msl" in result:
        result["msl"] = result["msl"] / 100
    if "d2m" in result:
        result["d2m"] = result["d2m"] - 273.15
    return result


def compute_wind_speed(ds):
    if "u10m" in ds and "v10m" in ds:
        ds["wind_speed"] = np.sqrt(ds["u10m"] ** 2 + ds["v10m"] ** 2)
    return ds


def compute_ensemble_stats(ds):
    if "member" not in ds.dims:
        return ds, None
    return ds.mean(dim="member"), ds.std(dim="member")


def get_city_forecast(ds_display, lead_time_hr):
    rows = []
    for city, (lat, lon) in PAKISTAN_CITIES.items():
        row = {"City": city}
        for var in ds_display.data_vars:
            info = VARIABLE_INFO.get(var, {"name": var, "unit": ""})
            try:
                val = ds_display[var].sel(lat=lat, lon=lon, method="nearest")
                if "lead_time" in val.dims:
                    target_lt = np.timedelta64(lead_time_hr, "h")
                    if target_lt in val.lead_time.values:
                        val = val.sel(lead_time=target_lt)
                    else:
                        val = val.isel(lead_time=0)
                row[f"{info['name']} ({info['unit']})"] = round(float(val.squeeze().values), 1)
            except Exception:
                row[f"{info['name']} ({info['unit']})"] = None
        rows.append(row)
    return rows


def get_city_timeseries(ds_display, city_name, var_key):
    if city_name not in PAKISTAN_CITIES:
        return None, None
    lat, lon = PAKISTAN_CITIES[city_name]
    try:
        val = ds_display[var_key].sel(lat=lat, lon=lon, method="nearest").squeeze()
        if "lead_time" in val.dims:
            hours = [int(lt / np.timedelta64(1, "h")) for lt in val.lead_time.values]
            values = val.values.tolist()
            return hours, values
    except Exception:
        pass
    return None, None


def check_alerts(ds_display, lead_time_hr):
    alerts = []
    for var, thresholds in ALERT_THRESHOLDS.items():
        if var not in ds_display:
            continue
        data = ds_display[var]
        if "lead_time" in data.dims:
            target_lt = np.timedelta64(lead_time_hr, "h")
            if target_lt in data.lead_time.values:
                data = data.sel(lead_time=target_lt)
            else:
                data = data.isel(lead_time=0)
        data = data.squeeze()
        max_val = float(data.max(skipna=True).values)
        min_val = float(data.min(skipna=True).values)
        if thresholds["high"] is not None and max_val > thresholds["high"]:
            alerts.append(("error", thresholds["high_label"], f"Max: {max_val:.1f}"))
        if thresholds["low"] is not None and min_val < thresholds["low"]:
            alerts.append(("warning", thresholds["low_label"], f"Min: {min_val:.1f}"))
    return alerts
