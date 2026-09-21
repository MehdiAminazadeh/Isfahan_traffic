from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import holidays
import jdatetime
import numpy as np
import pandas as pd
import requests

from .paths import PROJECT_ROOT, ensure_project_directories, load_settings


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OXCGRT_URL = (
    "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/"
    "data/OxCGRT_compact_national_v1.csv"
)
WEATHER_FIELDS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
)
JALALI_MONTHS = (
    "Farvardin",
    "Ordibehesht",
    "Khordad",
    "Tir",
    "Mordad",
    "Shahrivar",
    "Mehr",
    "Aban",
    "Azar",
    "Dey",
    "Bahman",
    "Esfand",
)


def _context_settings() -> Dict[str, object]:
    settings = load_settings()
    context = dict(settings.get("external_context", {}))
    context.setdefault("start_date", "2018-10-01")
    context.setdefault("end_date", "2021-01-31")
    context.setdefault("isfahan_latitude", 32.6546)
    context.setdefault("isfahan_longitude", 51.6680)
    context.setdefault("timezone", settings.get("timezone", "Asia/Tehran"))
    context.setdefault("weather_model", "era5")
    context["trusted_temporal_coverage"] = settings.get("trusted_temporal_coverage", 0.8)
    return context


def _date_window(settings: Dict[str, object]) -> pd.DatetimeIndex:
    start = pd.Timestamp(str(settings["start_date"])).normalize()
    end = pd.Timestamp(str(settings["end_date"])).normalize()
    if end < start:
        raise ValueError("external_context.end_date must be on or after start_date")
    return pd.date_range(start, end, freq="D")


def _iran_holidays(years: Iterable[int], language: str):
    try:
        return holidays.country_holidays("IR", years=list(years), language=language)
    except (TypeError, ValueError):
        return holidays.Iran(years=list(years), language=language)


def build_iran_calendar(
    dates: Sequence[pd.Timestamp], ramadan_windows: Optional[Sequence[Sequence[str]]] = None
) -> pd.DataFrame:
    date_index = pd.DatetimeIndex(dates).normalize()
    years = range(int(date_index.year.min()), int(date_index.year.max()) + 1)
    holiday_en = _iran_holidays(years, "en_US")
    holiday_fa = _iran_holidays(years, "fa")
    rows: List[Dict[str, object]] = []

    for timestamp in date_index:
        gregorian = timestamp.date()
        jalali = jdatetime.date.fromgregorian(date=gregorian)
        english_name = holiday_en.get(gregorian, "")
        persian_name = holiday_fa.get(gregorian, "")
        rows.append(
            {
                "date": timestamp,
                "weekday": int(timestamp.dayofweek),
                "weekday_name": timestamp.day_name(),
                "is_friday_weekend": int(timestamp.dayofweek == 4),
                "is_thursday": int(timestamp.dayofweek == 3),
                "is_official_holiday": int(bool(english_name or persian_name)),
                "official_holiday_name_en": english_name,
                "official_holiday_name_fa": persian_name,
                "jalali_year": int(jalali.year),
                "jalali_month": int(jalali.month),
                "jalali_day": int(jalali.day),
                "jalali_date": "%04d-%02d-%02d" % (jalali.year, jalali.month, jalali.day),
                "jalali_month_name_en": JALALI_MONTHS[jalali.month - 1],
                "is_nowruz_window": int(jalali.month == 1 and jalali.day <= 13),
                "is_nowruz_official_days": int(jalali.month == 1 and jalali.day <= 4),
            }
        )

    calendar = pd.DataFrame(rows)
    calendar["is_ramadan_approx"] = 0
    for bounds in ramadan_windows or ():
        start, end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        calendar.loc[calendar["date"].between(start, end), "is_ramadan_approx"] = 1

    holiday_positions = np.flatnonzero(calendar["is_official_holiday"].to_numpy() == 1)
    positions = np.arange(len(calendar))
    if len(holiday_positions):
        calendar["days_to_nearest_official_holiday"] = np.min(
            np.abs(positions[:, None] - holiday_positions[None, :]), axis=1
        )
    else:
        calendar["days_to_nearest_official_holiday"] = np.nan
    calendar["is_day_before_holiday"] = (
        calendar["is_official_holiday"].shift(-1, fill_value=0).astype(int)
    )
    calendar["is_day_after_holiday"] = (
        calendar["is_official_holiday"].shift(1, fill_value=0).astype(int)
    )
    calendar["is_nonworking_day"] = (
        (calendar["is_friday_weekend"] == 1) | (calendar["is_official_holiday"] == 1)
    ).astype(int)
    return calendar


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_to_path(url: str, target: Path, force: bool = False) -> Path:
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    temporary.replace(target)
    return target


def fetch_isfahan_weather(
    settings: Dict[str, object], dates: Sequence[pd.Timestamp], force: bool = False
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    model = str(settings["weather_model"])
    raw_path = PROJECT_ROOT / "data" / "external" / "raw" / ("open_meteo_%s_isfahan.json" % model)
    if force or not raw_path.exists():
        params = {
            "latitude": settings["isfahan_latitude"],
            "longitude": settings["isfahan_longitude"],
            "start_date": pd.DatetimeIndex(dates).min().strftime("%Y-%m-%d"),
            "end_date": pd.DatetimeIndex(dates).max().strftime("%Y-%m-%d"),
            "hourly": ",".join(WEATHER_FIELDS),
            "timezone": settings["timezone"],
            "models": settings["weather_model"],
        }
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=180)
        response.raise_for_status()
        #save the original bytes because requests can guess the wrong encoding
        raw_path.write_bytes(response.content)

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    hourly_payload = payload.get("hourly", {})
    missing = [field for field in ("time",) + WEATHER_FIELDS if field not in hourly_payload]
    if missing:
        raise ValueError("Open-Meteo response is missing fields: %s" % ", ".join(missing))
    hourly = pd.DataFrame({field: hourly_payload[field] for field in ("time",) + WEATHER_FIELDS})
    hourly = hourly.rename(columns={"time": "timestamp_local"})
    hourly["timestamp_local"] = pd.to_datetime(hourly["timestamp_local"], errors="raise")
    hourly["date"] = hourly["timestamp_local"].dt.normalize()
    empty_fields = [field for field in WEATHER_FIELDS if hourly[field].notna().sum() == 0]
    if empty_fields:
        raise ValueError(
            "Open-Meteo returned only null values for: %s. Try another model."
            % ", ".join(empty_fields)
        )

    hourly["source_grid_latitude"] = payload.get("latitude")
    hourly["source_grid_longitude"] = payload.get("longitude")
    hourly["source_grid_elevation_m"] = payload.get("elevation")
    hourly_path = (
        PROJECT_ROOT / "data" / "external" / "processed" / "isfahan_weather_hourly.csv.gz"
    )
    hourly.to_csv(hourly_path, index=False, compression="gzip")

    grouped = hourly.groupby("date", as_index=False)
    daily = grouped.agg(
        weather_hour_count=("timestamp_local", "size"),
        temperature_mean_c=("temperature_2m", "mean"),
        temperature_min_c=("temperature_2m", "min"),
        temperature_max_c=("temperature_2m", "max"),
        relative_humidity_mean_pct=("relative_humidity_2m", "mean"),
        precipitation_sum_mm=("precipitation", lambda values: values.sum(min_count=1)),
        rain_sum_mm=("rain", lambda values: values.sum(min_count=1)),
        snowfall_sum_cm=("snowfall", lambda values: values.sum(min_count=1)),
        cloud_cover_mean_pct=("cloud_cover", "mean"),
        wind_speed_mean_kmh=("wind_speed_10m", "mean"),
        wind_speed_max_kmh=("wind_speed_10m", "max"),
        wind_gust_max_kmh=("wind_gusts_10m", "max"),
    )
    wet_hours = (
        hourly.assign(wet=(hourly["precipitation"].fillna(0) >= 0.1).astype(int))
        .groupby("date", as_index=False)["wet"]
        .sum()
        .rename(columns={"wet": "wet_hours"})
    )
    daily = daily.merge(wet_hours, on="date", how="left")
    daily["is_heavy_precipitation_day"] = (daily["precipitation_sum_mm"] >= 10).astype(int)
    daily["is_wet_day"] = (daily["precipitation_sum_mm"] >= 0.1).astype(int)
    daily["is_hot_day"] = (daily["temperature_max_c"] >= 35).astype(int)
    daily["is_freezing_day"] = (daily["temperature_min_c"] <= 0).astype(int)
    metadata = {
        "url": OPEN_METEO_ARCHIVE_URL,
        "model": settings["weather_model"],
        "requested_coordinates": [settings["isfahan_latitude"], settings["isfahan_longitude"]],
        "returned_grid_coordinates": [payload.get("latitude"), payload.get("longitude")],
        "returned_elevation_m": payload.get("elevation"),
        "timezone": settings["timezone"],
        "hourly_units": payload.get("hourly_units", {}),
        "raw_sha256": _sha256(raw_path),
    }
    return daily, metadata


def fetch_iran_covid_policy(
    dates: Sequence[pd.Timestamp], force: bool = False
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    raw_path = PROJECT_ROOT / "data" / "external" / "raw" / "OxCGRT_compact_national_v1.csv"
    _download_to_path(OXCGRT_URL, raw_path, force=force)
    columns = [
        "CountryName", "CountryCode", "Jurisdiction", "Date",
        "C1M_School closing", "C1M_Flag", "C2M_Workplace closing", "C2M_Flag",
        "C3M_Cancel public events", "C3M_Flag", "C4M_Restrictions on gatherings", "C4M_Flag",
        "C5M_Close public transport", "C5M_Flag", "C6M_Stay at home requirements", "C6M_Flag",
        "C7M_Restrictions on internal movement", "C7M_Flag", "C8EV_International travel controls",
        "H6M_Facial Coverings", "StringencyIndex_Average", "GovernmentResponseIndex_Average",
        "ContainmentHealthIndex_Average", "EconomicSupportIndex", "ConfirmedCases", "ConfirmedDeaths",
    ]
    raw = pd.read_csv(raw_path, usecols=columns, low_memory=False)
    iran = raw[(raw["CountryCode"] == "IRN") & (raw["Jurisdiction"] == "NAT_TOTAL")].copy()
    iran["date"] = pd.to_datetime(iran["Date"].astype(str), format="%Y%m%d", errors="raise")
    iran = iran.drop_duplicates("date", keep="last")
    rename = {
        "C1M_School closing": "covid_school_closing_level",
        "C1M_Flag": "covid_school_closing_general_flag",
        "C2M_Workplace closing": "covid_workplace_closing_level",
        "C2M_Flag": "covid_workplace_closing_general_flag",
        "C3M_Cancel public events": "covid_public_event_cancel_level",
        "C3M_Flag": "covid_public_event_cancel_general_flag",
        "C4M_Restrictions on gatherings": "covid_gathering_restriction_level",
        "C4M_Flag": "covid_gathering_restriction_general_flag",
        "C5M_Close public transport": "covid_public_transport_closing_level",
        "C5M_Flag": "covid_public_transport_closing_general_flag",
        "C6M_Stay at home requirements": "covid_stay_home_level",
        "C6M_Flag": "covid_stay_home_general_flag",
        "C7M_Restrictions on internal movement": "covid_internal_movement_restriction_level",
        "C7M_Flag": "covid_internal_movement_restriction_general_flag",
        "C8EV_International travel controls": "covid_international_travel_control_level",
        "H6M_Facial Coverings": "covid_facial_covering_level",
        "StringencyIndex_Average": "covid_stringency_index",
        "GovernmentResponseIndex_Average": "covid_government_response_index",
        "ContainmentHealthIndex_Average": "covid_containment_health_index",
        "EconomicSupportIndex": "covid_economic_support_index",
        "ConfirmedCases": "covid_confirmed_cases_cumulative",
        "ConfirmedDeaths": "covid_confirmed_deaths_cumulative",
    }
    iran = iran.rename(columns=rename)
    keep = ["date"] + list(rename.values())
    iran = iran[keep]

    output = pd.DataFrame({"date": pd.DatetimeIndex(dates).normalize()})
    output = output.merge(iran, on="date", how="left", indicator="_covid_merge")
    output["covid_data_available"] = (output["_covid_merge"] == "both").astype(int)
    output = output.drop(columns="_covid_merge")
    policy_columns = [column for column in rename.values() if "confirmed_" not in column]
    first_available = iran["date"].min()
    #treat dates before the policy series as having no COVID restrictions
    #leave gaps within the series missing
    pre_source = output["date"] < first_available
    output.loc[pre_source, policy_columns] = output.loc[pre_source, policy_columns].fillna(0)
    for column in ("covid_confirmed_cases_cumulative", "covid_confirmed_deaths_cumulative"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
        output.loc[pre_source, column] = 0
        output[column] = output[column].ffill()
    output["covid_new_cases_derived"] = output["covid_confirmed_cases_cumulative"].diff().fillna(0)
    output["covid_new_deaths_derived"] = output["covid_confirmed_deaths_cumulative"].diff().fillna(0)
    output["covid_any_mobility_restriction"] = (
        output[
            [
                "covid_workplace_closing_level",
                "covid_public_transport_closing_level",
                "covid_stay_home_level",
                "covid_internal_movement_restriction_level",
            ]
        ].fillna(0).max(axis=1) > 0
    ).astype(int)
    metadata = {
        "url": OXCGRT_URL,
        "country_code": "IRN",
        "jurisdiction": "NAT_TOTAL",
        "scope_warning": "National policy values; they are not Isfahan-specific compliance or mobility measurements.",
        "raw_sha256": _sha256(raw_path),
        "source_first_date": first_available.strftime("%Y-%m-%d"),
        "source_last_date": iran["date"].max().strftime("%Y-%m-%d"),
    }
    return output, metadata


def load_manual_events(
    dates: Sequence[pd.Timestamp], path: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    event_path = path or (
        PROJECT_ROOT / "data" / "external" / "manual" / "isfahan_events.csv"
    )
    events = pd.read_csv(event_path)
    events["date_start"] = pd.to_datetime(events["date_start"], errors="raise")
    events["date_end"] = pd.to_datetime(events["date_end"], errors="raise")
    if (events["date_end"] < events["date_start"]).any():
        raise ValueError("manual event date_end cannot be earlier than date_start")
    wanted = set(pd.DatetimeIndex(dates).normalize())
    expanded: List[Dict[str, object]] = []
    confidence_rank = {"low": 1, "medium": 2, "high": 3}
    for row in events.itertuples(index=False):
        for date in pd.date_range(row.date_start, row.date_end, freq="D"):
            if date in wanted:
                expanded.append(
                    {
                        "date": date,
                        "event_type": row.event_type,
                        "event_name": row.event_name,
                        "source_confidence": row.source_confidence,
                        "confidence_rank": confidence_rank.get(str(row.source_confidence).lower(), 0),
                    }
                )
    daily = pd.DataFrame({"date": pd.DatetimeIndex(dates).normalize()})
    if expanded:
        event_days = pd.DataFrame(expanded)
        grouped = event_days.groupby("date", as_index=False).agg(
            manual_event_count=("event_name", "size"),
            manual_event_types=("event_type", lambda values: " | ".join(sorted(set(values)))),
            manual_event_names=("event_name", lambda values: " | ".join(sorted(set(values)))),
            manual_event_confidence_rank=("confidence_rank", "max"),
        )
        daily = daily.merge(grouped, on="date", how="left")
    #create the columns even when the event registry is empty
    defaults = {
        "manual_event_count": 0,
        "manual_event_types": "",
        "manual_event_names": "",
        "manual_event_confidence_rank": 0,
    }
    for column, default in defaults.items():
        if column not in daily:
            daily[column] = default
        daily[column] = daily[column].fillna(default)
    daily["manual_event_count"] = daily["manual_event_count"].astype(int)
    daily["manual_event_confidence_rank"] = daily["manual_event_confidence_rank"].astype(int)
    return daily, events


def _write_context_associations(city_context: pd.DataFrame, table_dir: Path) -> int:
    observed = city_context[city_context["city_median_15min_volume"].notna()].copy()
    if observed.empty:
        return 0
    observed["year_month"] = observed["date"].dt.strftime("%Y-%m")
    observed["expected_same_month_weekday_volume"] = observed.groupby(
        ["year_month", "weekday"]
    )["city_median_15min_volume"].transform("median")
    observed["expected_same_month_volume"] = observed.groupby("year_month")[
        "city_median_15min_volume"
    ].transform("median")
    observed["month_adjusted_volume_index"] = (
        100.0 * observed["city_median_15min_volume"] / observed["expected_same_month_volume"]
    )
    observed["adjusted_volume_index"] = (
        100.0
        * observed["city_median_15min_volume"]
        / observed["expected_same_month_weekday_volume"]
    )
    factor_columns = {
        "official_holiday": "is_official_holiday",
        "friday_weekend": "is_friday_weekend",
        "nowruz_window": "is_nowruz_window",
        "ramadan_approx": "is_ramadan_approx",
        "wet_weather": "is_wet_day",
        "heavy_precipitation": "is_heavy_precipitation_day",
        "hot_day_35c": "is_hot_day",
        "freezing_day": "is_freezing_day",
        "covid_mobility_restriction": "covid_any_mobility_restriction",
        "documented_manual_event": "manual_event_count",
    }
    summaries: List[Dict[str, object]] = []
    for factor, column in factor_columns.items():
        flagged = observed[column].fillna(0) > 0
        with_factor = observed.loc[flagged]
        without_factor = observed.loc[~flagged]
        summaries.append(
            {
                "context_factor": factor,
                "flagged_days_with_traffic": int(len(with_factor)),
                "comparison_days_with_traffic": int(len(without_factor)),
                "flagged_mean_adjusted_volume_index": with_factor["adjusted_volume_index"].mean(),
                "flagged_median_adjusted_volume_index": with_factor["adjusted_volume_index"].median(),
                "comparison_mean_adjusted_volume_index": without_factor["adjusted_volume_index"].mean(),
                "adjusted_index_difference_points": (
                    with_factor["adjusted_volume_index"].mean()
                    - without_factor["adjusted_volume_index"].mean()
                ),
                "flagged_mean_month_adjusted_volume_index": with_factor[
                    "month_adjusted_volume_index"
                ].mean(),
                "comparison_mean_month_adjusted_volume_index": without_factor[
                    "month_adjusted_volume_index"
                ].mean(),
                "month_adjusted_index_difference_points": (
                    with_factor["month_adjusted_volume_index"].mean()
                    - without_factor["month_adjusted_volume_index"].mean()
                ),
                "flagged_mean_raw_15min_volume": with_factor["city_median_15min_volume"].mean(),
                "comparison_mean_raw_15min_volume": without_factor["city_median_15min_volume"].mean(),
                "interpretation": (
                    "descriptive association only; adjusted_volume_index controls for Gregorian month "
                    "and weekday, while month_adjusted_volume_index retains weekday effects"
                ),
            }
        )
    pd.DataFrame(summaries).to_csv(table_dir / "traffic_context_associations.csv", index=False)
    return int(len(summaries))


def _build_analysis_ready_joins(context: pd.DataFrame, settings: Dict[str, object]) -> Dict[str, int]:
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    traffic_path = table_dir / "intersection_daily.csv"
    result = {
        "intersection_context_rows": 0,
        "city_context_rows": 0,
        "city_traffic_available_days": 0,
        "city_traffic_missing_days": 0,
        "association_rows": 0,
    }
    if not traffic_path.exists():
        return result
    traffic = pd.read_csv(traffic_path, parse_dates=["date"])
    joined = traffic.merge(context, on="date", how="left", validate="many_to_one")
    joined.to_csv(table_dir / "intersection_daily_with_context.csv.gz", index=False, compression="gzip")
    result["intersection_context_rows"] = int(len(joined))

    trusted = traffic[
        traffic["temporal_coverage"] >= float(settings["trusted_temporal_coverage"])
    ].copy()
    city = trusted.groupby("date", as_index=False).agg(
        trusted_intersection_count=("intersection_id", "nunique"),
        city_median_15min_volume=("mean_15min_volume", "median"),
        city_mean_15min_volume=("mean_15min_volume", "mean"),
        city_total_volume_estimate=("total_volume_estimate", "sum"),
    )
    city_context = context.merge(city, on="date", how="left", validate="one_to_one")
    city_context["traffic_data_available"] = city_context["city_median_15min_volume"].notna().astype(int)
    city_context.to_csv(table_dir / "city_daily_with_context.csv", index=False)
    result["city_context_rows"] = int(len(city_context))
    result["city_traffic_available_days"] = int(city_context["traffic_data_available"].sum())
    result["city_traffic_missing_days"] = int((city_context["traffic_data_available"] == 0).sum())
    result["association_rows"] = _write_context_associations(city_context, table_dir)
    return result


def _build_monthly_context(context: pd.DataFrame) -> pd.DataFrame:
    work = context.copy()
    work["year_month"] = work["date"].dt.strftime("%Y-%m")
    return work.groupby("year_month", as_index=False).agg(
        calendar_days=("date", "size"),
        official_holiday_days=("is_official_holiday", "sum"),
        friday_days=("is_friday_weekend", "sum"),
        nowruz_window_days=("is_nowruz_window", "sum"),
        ramadan_approx_days=("is_ramadan_approx", "sum"),
        manual_event_days=("manual_event_count", lambda values: int((values > 0).sum())),
        temperature_mean_c=("temperature_mean_c", "mean"),
        temperature_max_c=("temperature_max_c", "max"),
        precipitation_sum_mm=("precipitation_sum_mm", "sum"),
        heavy_precipitation_days=("is_heavy_precipitation_day", "sum"),
        covid_stringency_mean=("covid_stringency_index", "mean"),
        covid_stringency_max=("covid_stringency_index", "max"),
        covid_mobility_restriction_days=("covid_any_mobility_restriction", "sum"),
    )


def build_external_context(force: bool = False) -> Dict[str, object]:
    ensure_project_directories()
    settings = _context_settings()
    dates = _date_window(settings)
    calendar = build_iran_calendar(dates, settings.get("ramadan_approximate_windows", []))
    weather, weather_meta = fetch_isfahan_weather(settings, dates, force=force)
    covid, covid_meta = fetch_iran_covid_policy(dates, force=force)
    event_daily, events = load_manual_events(dates)

    context = calendar.merge(weather, on="date", how="left", validate="one_to_one")
    context = context.merge(covid, on="date", how="left", validate="one_to_one")
    context = context.merge(event_daily, on="date", how="left", validate="one_to_one")
    processed_dir = PROJECT_ROOT / "data" / "external" / "processed"
    table_dir = PROJECT_ROOT / "outputs" / "tables"
    daily_path = processed_dir / "isfahan_daily_context.csv"
    context.to_csv(daily_path, index=False, date_format="%Y-%m-%d")
    monthly = _build_monthly_context(context)
    monthly.to_csv(table_dir / "external_context_monthly.csv", index=False)
    join_counts = _build_analysis_ready_joins(context, settings)

    core_weather_columns = [
        "temperature_mean_c",
        "relative_humidity_mean_pct",
        "precipitation_sum_mm",
        "cloud_cover_mean_pct",
        "wind_speed_mean_kmh",
    ]
    weather_missing_by_field = {
        column: int(context[column].isna().sum()) for column in core_weather_columns
    }
    weather_missing = int(context[core_weather_columns].isna().any(axis=1).sum())
    covid_expected = context["date"] >= pd.Timestamp(covid_meta["source_first_date"])
    covid_missing = int(context.loc[covid_expected, "covid_data_available"].eq(0).sum())
    quality: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_start": dates.min().strftime("%Y-%m-%d"),
        "date_end": dates.max().strftime("%Y-%m-%d"),
        "expected_daily_rows": int(len(dates)),
        "daily_rows": int(len(context)),
        "duplicate_dates": int(context["date"].duplicated().sum()),
        "weather_missing_days": weather_missing,
        "weather_missing_by_field": weather_missing_by_field,
        "covid_missing_days_inside_source_window": covid_missing,
        "official_holiday_days": int(context["is_official_holiday"].sum()),
        "manual_event_records": int(len(events)),
        "manual_event_days": int((context["manual_event_count"] > 0).sum()),
        "traffic_join": join_counts,
        "passed": bool(
            len(context) == len(dates)
            and context["date"].duplicated().sum() == 0
            and weather_missing == 0
            and covid_missing == 0
        ),
        "caveats": [
            "ERA5 is gridded reanalysis at the nearest model cell, not an intersection sensor.",
            "OxCGRT policy indicators are national; local enforcement and compliance can differ in Isfahan.",
            "Ramadan windows are approximate civil-calendar windows and can differ by one day from observation.",
            "An event-context match is an explanatory candidate, not proof of causality.",
        ],
    }
    (table_dir / "external_context_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "retrieved_at_utc": quality["generated_at_utc"],
        "calendar": {
            "provider": "Vacanza holidays 0.57 / Iran calendar",
            "documentation": "https://holidays.readthedocs.io/en/latest/auto_gen_docs/iran/",
            "verification_sites": ["https://www.time.ir/", "https://calendar.ut.ac.ir/"],
            "scope": "Iran national public holidays; Persian and English labels",
        },
        "weather": weather_meta,
        "covid_policy": covid_meta,
        "manual_events_file": str(
            (PROJECT_ROOT / "data" / "external" / "manual" / "isfahan_events.csv").relative_to(PROJECT_ROOT)
        ),
    }
    (PROJECT_ROOT / "data" / "external" / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "External context: %d days, %d holidays, %d event-days, weather gaps=%d, QA=%s"
        % (
            len(context),
            context["is_official_holiday"].sum(),
            (context["manual_event_count"] > 0).sum(),
            weather_missing,
            "PASS" if quality["passed"] else "CHECK",
        )
    )
    return quality
