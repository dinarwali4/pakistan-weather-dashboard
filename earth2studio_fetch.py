"""
earth2studio_fetch.py
Fetches GFS weather data directly from NOAA via plain HTTP.
No async, no S3 client, no event loop issues. Works everywhere.
"""

import numpy as np
import xarray as xr
import streamlit as st
import os
import hashlib
import tempfile
import urllib.request
from datetime import datetime, timedelta

# Pakistan region bounds
PK_LAT_MIN = 24.0
PK_LAT_MAX = 37.0
PK_LON_MIN = 61.0
PK_LON_MAX = 77.0

# GFS index patterns for each variable
GFS_IDX_PATTERNS = {
    "t2m": "TMP:2 m above ground",
    "tp": "APCP:surface",
    "u10m": "UGRD:10 m above ground",
    "v10m": "VGRD:10 m above ground",
    "r2m": "RH:2 m above ground",
    "msl": "PRMSL:mean sea level",
}

VARIABLE_INFO = {
    "t2m": {"name": "2m Temperature", "unit": "°C", "cmap": "RdBu_r"},
    "tp": {"name": "Precipitation", "unit": "mm", "cmap": "Blues"},
    "wind_speed": {"name": "Wind Speed", "unit": "m/s", "cmap": "YlOrRd"},
    "r2m": {"name": "Relative Humidity", "unit": "%", "cmap": "YlGn"},
    "msl": {"name": "Sea Level Pressure", "unit": "hPa", "cmap": "coolwarm"},
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

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _get_cache_path(key):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode()).hexdigest() + ".nc")


# -----------------------------------------------------------
# Direct HTTP fetch — no async, no S3, no event loop issues
# -----------------------------------------------------------

def _gfs_base_url(init_date, lead_hour):
    return (
        f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        f"gfs.{init_date}/00/atmos/gfs.t00z.pgrb2.0p25.f{lead_hour:03d}"
    )


def _fetch_idx(url):
    """Download and parse GFS .idx index file to get byte offsets."""
    req = urllib.request.Request(url + ".idx")
    with urllib.request.urlopen(req, timeout=30) as resp:
        lines = resp.read().decode("utf-8").strip().split("\n")

    entries = []
    for line in lines:
        parts = line.split(":")
        if len(parts) >= 7:
            entries.append({
                "byte_start": int(parts[1]),
                "key": f"{parts[3]}:{parts[4]}",
            })

    # Compute byte_end for each entry
    for i in range(len(entries) - 1):
        entries[i]["byte_end"] = entries[i + 1]["byte_start"] - 1
    if entries:
        entries[-1]["byte_end"] = None  # last goes to end
    return entries


def _download_variable(url, byte_start, byte_end):
    """Download a single GRIB variable via HTTP Range request."""
    if byte_end is None:
        range_hdr = f"bytes={byte_start}-"
    else:
        range_hdr = f"bytes={byte_start}-{byte_end}"
    req = urllib.request.Request(url, headers={"Range": range_hdr})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _parse_grib_to_subset(grib_bytes):
    """Parse GRIB bytes and extract Pakistan region as numpy array."""
    import pygrib

    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(grib_bytes)
        tmp = f.name
    try:
        grbs = pygrib.open(tmp)
        msg = grbs[1]
        lats_2d, lons_2d = msg.latlons()
        values = msg.values

        lat_1d = lats_2d[:, 0]
        lon_1d = lons_2d[0, :]

        lat_mask = (lat_1d >= PK_LAT_MIN) & (lat_1d <= PK_LAT_MAX)
        lon_mask = (lon_1d >= PK_LON_MIN) & (lon_1d <= PK_LON_MAX)

        subset = values[np.ix_(lat_mask, lon_mask)]
        grbs.close()
        return subset, lat_1d[lat_mask], lon_1d[lon_mask]
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_gfs_forecast(init_time, lead_hours=None, variables=None):
    """Fetch GFS data via plain HTTP. No async. Works on Streamlit Cloud."""
    if lead_hours is None:
        lead_hours = [24]
    if variables is None:
        variables = list(GFS_IDX_PATTERNS.keys())

    init_date = init_time.split("T")[0].replace("-", "")
    cache_key = f"gfs_{init_date}_h{'_'.join(map(str, lead_hours))}_v{'_'.join(variables)}"
    cache_path = _get_cache_path(cache_key)

    if os.path.exists(cache_path):
        try:
            return xr.open_dataset(cache_path)
        except Exception:
            pass

    all_data = {}
    ref_lats = None
    ref_lons = None

    for lead_hr in lead_hours:
        url = _gfs_base_url(init_date, lead_hr)

        try:
            idx = _fetch_idx(url)
        except Exception as e:
            st.error(f"Cannot reach NOAA: {e}")
            return None

        for var_key in variables:
            pattern = GFS_IDX_PATTERNS.get(var_key)
            if not pattern:
                continue

            # Find matching entry in index
            match = None
            for entry in idx:
                if entry["key"] == pattern:
                    match = entry
                    break
            if match is None:
                continue

            try:
                grib_bytes = _download_variable(url, match["byte_start"], match["byte_end"])
                arr, lats, lons = _parse_grib_to_subset(grib_bytes)
                if ref_lats is None:
                    ref_lats = lats
                    ref_lons = lons
                all_data.setdefault(var_key, []).append(arr)
            except Exception as e:
                st.warning(f"Skip {var_key} +{lead_hr}h: {e}")

    if not all_data or ref_lats is None:
        st.error("No data could be fetched from NOAA.")
        return None

    lead_arr = np.array([np.timedelta64(h, "h") for h in lead_hours])
    data_vars = {}
    for var_key, arrays in all_data.items():
        data_vars[var_key] = (["lead_time", "lat", "lon"], np.stack(arrays, axis=0))

    ds = xr.Dataset(data_vars, coords={"lead_time": lead_arr, "lat": ref_lats, "lon": ref_lons})

    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        ds.to_netcdf(cache_path)
    except Exception:
        pass

    return ds


# --- Processing ---

def convert_units(ds):
    result = ds.copy()
    if "t2m" in result:
        result["t2m"] = result["t2m"] - 273.15
    if "msl" in result:
        result["msl"] = result["msl"] / 100
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
    # Only show meaningful variables, not raw u10m/v10m
    show_vars = ["t2m", "tp", "wind_speed", "r2m", "msl"]
    rows = []
    for city, (lat, lon) in PAKISTAN_CITIES.items():
        row = {"City": city}
        for var in show_vars:
            if var not in ds_display:
                continue
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
            return hours, val.values.tolist()
    except Exception:
        pass
    return None, None


def check_alerts(ds_display, lead_time_hr):
    alerts = []
    for var, th in ALERT_THRESHOLDS.items():
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
        mx = float(data.max(skipna=True).values)
        mn = float(data.min(skipna=True).values)
        if th["high"] is not None and mx > th["high"]:
            alerts.append(("error", th["high_label"], f"Max: {mx:.1f}"))
        if th["low"] is not None and mn < th["low"]:
            alerts.append(("warning", th["low_label"], f"Min: {mn:.1f}"))
    return alerts
