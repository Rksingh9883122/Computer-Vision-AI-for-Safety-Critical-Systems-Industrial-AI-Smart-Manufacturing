from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional: autorefresh component (preferred if installed)
try:
    from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
    _HAS_AUTOREFRESH = True
except Exception:
    _HAS_AUTOREFRESH = False

# -------------------------------
# Config
# -------------------------------
DEFAULT_LOG_DIR = Path(os.getenv("LOG_DIR", "logs")).resolve()
st.set_page_config(page_title="HSE Analytics", page_icon="📊", layout="wide")

# -------------------------------
# Helpers
# -------------------------------
@st.cache_data(show_spinner=False)
def list_detection_csvs(log_dir: Path) -> List[Path]:
    """List detection log CSV files, newest first."""
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob("detections_*.csv"), key=os.path.getmtime, reverse=True)


@st.cache_data(show_spinner=False)
def read_csvs(csv_paths: List[Path], fingerprint: Tuple[Tuple[str, int, int], ...]) -> pd.DataFrame:
    """
    Read and concatenate selected CSVs.

    NOTE:
    - `fingerprint` is not used directly inside the function. It's included
      solely to make Streamlit's cache re-run whenever any file's
      name/mtime/size changes (instant cache bust when files update).
    """
    dfs: List[pd.DataFrame] = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
            df["__source_csv__"] = p.name
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = {"server_time_iso", "class_name", "confidence"}
    for c in required - set(df.columns):
        df[c] = np.nan
    if "camera_id" not in df.columns:
        df["camera_id"] = ""
    if "location" not in df.columns:
        df["location"] = ""
    return df


def normalize_and_filter(
    df: pd.DataFrame,
    min_conf: float,
    tz_offset_min: int,
    date_start: pd.Timestamp | None,
    date_end: pd.Timestamp | None,
    camera_contains: str,
    location_contains: str,
    include_classes: List[str],
    exclude_classes: List[str],
) -> pd.DataFrame:

    # Parse timestamps and sort
    df["ts_utc"] = pd.to_datetime(df["server_time_iso"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"])

    # Numeric confidence
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

    # Apply min confidence
    df = df[df["confidence"] >= float(min_conf)]

    # Local time with tz offset
    df["ts_local"] = df["ts_utc"] + pd.to_timedelta(int(tz_offset_min), unit="m")

    # Date range
    if date_start is not None:
        df = df[df["ts_local"] >= pd.to_datetime(date_start)]
    if date_end is not None:
        end = pd.to_datetime(date_end)
        if end.time() == pd.Timestamp.min.time():
            end = end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df["ts_local"] <= end]

    # Camera/location contains
    if camera_contains.strip():
        df = df[df["camera_id"].astype(str).str.contains(camera_contains, case=False, na=False)]
    if location_contains.strip():
        df = df[df["location"].astype(str).str.contains(location_contains, case=False, na=False)]

    # Include/exclude classes
    if include_classes:
        df = df[df["class_name"].astype(str).isin(include_classes)]
    if exclude_classes:
        df = df[~df["class_name"].astype(str).isin(exclude_classes)]

    return df


def compute_aggregations(df: pd.DataFrame, top_k: int, stacked: bool) -> Dict[str, pd.DataFrame]:
    if df.empty:
        return {
            "hourly_counts": pd.DataFrame(columns=["hour", "count"]),
            "daily_counts": pd.DataFrame(columns=["date", "count"]),
            "class_counts": pd.DataFrame(columns=["class_name", "count"]),
            "daily_cat": pd.DataFrame(),
            "heatmap_hour_class": pd.DataFrame(),
        }

    # Hourly
    df["hour"] = df["ts_local"].dt.floor(pd.Timedelta(hours=1))
    hourly_counts = df.groupby("hour").size().reset_index(name="count")

    # Daily
    df["date"] = df["ts_local"].dt.date
    daily_counts = df.groupby("date").size().reset_index(name="count")

    # Top classes
    class_counts = df["class_name"].value_counts().head(top_k).reset_index()
    class_counts.columns = ["class_name", "count"]

    # Daily by class
    daily_cat = df.groupby(["date", "class_name"]).size().unstack(fill_value=0)

    if stacked and not daily_cat.empty and daily_cat.shape[1] > top_k:
        top_classes = class_counts["class_name"].tolist()
        keep = [c for c in daily_cat.columns if c in top_classes]
        other_cols = [c for c in daily_cat.columns if c not in keep]
        if other_cols:
            daily_cat["Other"] = daily_cat[other_cols].sum(axis=1)
        daily_cat = daily_cat[keep + (["Other"] if "Other" in daily_cat.columns else [])]

    # Heatmap: class vs hour (use top classes for readability)
    top_classes = class_counts["class_name"].tolist()
    df_top = df[df["class_name"].isin(top_classes)].copy()
    if df_top.empty:
        heatmap_hour_class = pd.DataFrame()
    else:
        df_top["hour"] = df_top["ts_local"].dt.floor(pd.Timedelta(hours=1))
        heatmap_hour_class = df_top.groupby([df_top["hour"], df_top["class_name"]]).size().unstack(fill_value=0).T

    return {
        "hourly_counts": hourly_counts,
        "daily_counts": daily_counts,
        "class_counts": class_counts,
        "daily_cat": daily_cat,
        "heatmap_hour_class": heatmap_hour_class,
    }

# -------------------------------
# UI - Sidebar
# -------------------------------
st.sidebar.title("⚙️ Controls")

log_dir = Path(st.sidebar.text_input("Logs directory", str(DEFAULT_LOG_DIR))).resolve()
all_csv_paths = list_detection_csvs(log_dir)

if not all_csv_paths:
    st.sidebar.warning(f"No CSV files found in: {log_dir}")
else:
    st.sidebar.caption(f"Found {len(all_csv_paths)} log file(s)")

csv_options = ["All CSVs"] + [p.name for p in all_csv_paths]
selected_csvs = st.sidebar.multiselect("Select CSV(s)", csv_options, default=["All CSVs"])

# Filters
min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.25, 0.01)
tz_offset_min = st.sidebar.number_input("TZ Offset (minutes)", min_value=-12*60, max_value=14*60, value=0, step=30)

date_range = st.sidebar.date_input("Date range (local time)", [])
date_start = date_range[0] if isinstance(date_range, (list, tuple)) and len(date_range) >= 1 else None
date_end = date_range[-1] if isinstance(date_range, (list, tuple)) and len(date_range) == 2 else None

camera_contains = st.sidebar.text_input("Camera ID contains", "")
location_contains = st.sidebar.text_input("Location contains", "")

# Include/Exclude classes are populated after we load DF
stacked = st.sidebar.toggle("Stack daily by class", value=True)
top_k = st.sidebar.slider("Top-K classes", 1, 30, 15)

# Page auto-refresh
auto_refresh = st.sidebar.selectbox("Auto-refresh", ["Off", "10 sec", "30 sec", "60 sec"], index=0)
refresh_map = {"Off": 0, "10 sec": 10_000, "30 sec": 30_000, "60 sec": 60_000}
interval_ms = refresh_map.get(auto_refresh, 0)

if interval_ms > 0:
    # 1) Bust page-level cache/state via query params (new API + fallback)
    token = str(np.random.randint(0, 999_999))
    try:
        st.query_params["_"] = token  # values must be strings
    except Exception:
        if hasattr(st, "experimental_set_query_params"):
            st.experimental_set_query_params(_=token)

    # 2) Trigger periodic reruns
    if _HAS_AUTOREFRESH:
        st_autorefresh(interval=interval_ms, key="hse_autorefresh")
    else:
        seconds = max(int(interval_ms // 1000), 1)
        try:
            st.html(f'<meta http-equiv="refresh" content="{seconds}">')
        except Exception:
            st.markdown(f'<meta http-equiv="refresh" content="{seconds}">', unsafe_allow_html=True)

# -------------------------------
# Load data (with deterministic cache-busting)
# -------------------------------
if "All CSVs" in selected_csvs or not selected_csvs:
    csv_paths = all_csv_paths
else:
    csv_paths = [p for p in all_csv_paths if p.name in selected_csvs]

# Build fingerprint of files (name, mtime, size) → any change busts the cache immediately
fingerprint: Tuple[Tuple[str, int, int], ...] = tuple(
    (p.name, int(p.stat().st_mtime), p.stat().st_size)
    for p in csv_paths
)

df_raw = read_csvs(csv_paths, fingerprint=fingerprint)
df_raw = ensure_columns(df_raw)

st.title("📊 HSE Analytics")
st.caption(f"Source directory: `{log_dir}`")

if df_raw.empty:
    st.info("No data available yet. Start your streams or upload some logs.")
    st.stop()

# Populate include/exclude class pickers with available classes
all_classes = sorted(pd.Series(df_raw["class_name"].astype(str).unique()).dropna().tolist())
with st.sidebar.expander("Class filters"):
    include_classes = st.multiselect("Include only", options=all_classes, default=[])
    exclude_classes = st.multiselect("Exclude", options=[c for c in all_classes if c not in include_classes], default=[])

# -------------------------------
# Filtered data
# -------------------------------
df = normalize_and_filter(
    df_raw.copy(),
    min_conf=min_conf,
    tz_offset_min=tz_offset_min,
    date_start=pd.to_datetime(date_start) if date_start else None,
    date_end=pd.to_datetime(date_end) if date_end else None,
    camera_contains=camera_contains,
    location_contains=location_contains,
    include_classes=include_classes,
    exclude_classes=exclude_classes,
)

if df.empty:
    st.warning("No rows after filters. Try relaxing the filters.")
    st.stop()

# -------------------------------
# KPIs
# -------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total observations", f"{df.shape[0]:,}")
col2.metric("Unique classes", df["class_name"].nunique())
col3.metric("Cameras", df["camera_id"].astype(str).nunique())
col4.metric("Locations", df["location"].astype(str).nunique())
span_txt = f"{pd.to_datetime(df['ts_local']).min():%Y-%m-%d %H:%M} → {pd.to_datetime(df['ts_local']).max():%Y-%m-%d %H:%M}"
col5.metric("Time span (local)", span_txt)

# -------------------------------
# Aggregations
# -------------------------------
aggs = compute_aggregations(df.copy(), top_k=top_k, stacked=stacked)
hourly_counts = aggs["hourly_counts"]
daily_counts = aggs["daily_counts"]
class_counts = aggs["class_counts"]
daily_cat = aggs["daily_cat"]
heatmap_hour_class = aggs["heatmap_hour_class"]

# -------------------------------
# Charts
# -------------------------------
st.subheader("Time-based views")

c1, c2 = st.columns(2)
with c1:
    fig_hourly = px.bar(
        hourly_counts, x="hour", y="count",
        title=f"Observations per Hour (TZ offset {tz_offset_min} min)",
        labels={"hour": "Hour", "count": "Count"}
    )
    fig_hourly.update_layout(hovermode="x unified", xaxis_tickformat="%Y-%m-%d %H:%M")
    st.plotly_chart(fig_hourly, use_container_width=True)

with c2:
    fig_daily = px.bar(
        daily_counts, x="date", y="count",
        title="Observations per Day",
        labels={"date": "Date", "count": "Count"}
    )
    fig_daily.update_layout(hovermode="x unified")
    st.plotly_chart(fig_daily, use_container_width=True)

fig_trend = px.line(
    daily_counts, x="date", y="count", markers=True,
    title="Trend of Observations Over Time", labels={"date": "Date", "count": "Count"}
)
st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("Class-based views")
c3, c4 = st.columns(2)
with c3:
    fig_top = px.bar(
        class_counts, x="class_name", y="count",
        title=f"Top {top_k} Classes",
        labels={"class_name": "Class", "count": "Count"}
    )
    fig_top.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig_top, use_container_width=True)

with c4:
    if not class_counts.empty:
        fig_pie = px.pie(class_counts, names="class_name", values="count", title="Class Distribution")
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No class data to show pie chart.")

st.subheader(f"Daily Counts by Class{' (stacked)' if stacked else ''}")
if not daily_cat.empty:
    df_long = daily_cat.copy().reset_index().rename(columns={"index": "date"})
    df_long = df_long.melt(id_vars=["date"], var_name="class_name", value_name="count")
    fig_daily_cat = px.bar(
        df_long, x="date", y="count", color="class_name",
        title=f"Daily Counts by Class{' (stacked)' if stacked else ''}",
        labels={"date": "Date", "count": "Count", "class_name": "Class"}
    )
    fig_daily_cat.update_layout(barmode="stack" if stacked else "group")
    st.plotly_chart(fig_daily_cat, use_container_width=True)
else:
    st.info("No data for daily-by-class chart.")

st.subheader("Class vs Hour Heatmap")
if not heatmap_hour_class.empty:
    fig_hm = go.Figure(
        data=go.Heatmap(
            z=heatmap_hour_class.values,
            x=[str(c) for c in heatmap_hour_class.columns],
            y=[str(i) for i in heatmap_hour_class.index],
            colorscale="YlOrRd",
            colorbar=dict(title="Count"),
        )
    )
    fig_hm.update_layout(title="Class vs Hour Heatmap", xaxis_title="Hour", yaxis_title="Class")
    st.plotly_chart(fig_hm, use_container_width=True)
else:
    st.info("No heatmap data (likely due to filters or no top classes).")

# -------------------------------
# Data Preview & Export
# -------------------------------
st.subheader("Filtered Data")
preview_cols = [
    "ts_utc", "ts_local", "camera_id", "location", "class_name", "confidence", "__source_csv__"
] if "__source_csv__" in df.columns else [
    "ts_utc", "ts_local", "camera_id", "location", "class_name", "confidence"
]
st.dataframe(
    df[preview_cols].sort_values("ts_local", ascending=False).head(1000),
    use_container_width=True,
    height=420
)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download filtered CSV", data=csv_bytes, file_name="filtered_observations.csv", mime="text/csv")

st.caption("Tip: Use auto-refresh (sidebar) to keep this dashboard updated while streams are running.")