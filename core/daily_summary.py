import os
from datetime import datetime, date
from typing import Optional

import pandas as pd


SUMMARY_COLUMNS = [
    "Date",
    "Day",
    "Approved Strength",
    "Required Strength",
    "Onroll",
    "Contractor",
    "App",
    "C.Off",
    "Total (Onroll Strength)",
    "OJT",
    "Onroll Present",
    "Contractor Present",
    "App Present",
    "Total Present",
    "OJT Present",
    "Extra Hours",
    "Extra Hours Mandays",
    "Total Manpower Utilized",
    "Gap",
    "Cumulative Mandays",
    "Manpower Shortage % (Approved)",
    "Manpower Shortage % (Required)",
    "Absenteeism Onroll",
    "Absenteeism Contractor",
    "Absenteeism App",
    "Rolling Shift Week Off",
    "Informed Leave",
]


def _safe_round(value, ndigits=2):
    if value is None or pd.isna(value):
        return "NA"
    return round(value, ndigits)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if "category" in out.columns:
        out["category"] = out["category"].astype(str).str.lower().str.strip()
    if "status" in out.columns:
        out["status"] = out["status"].astype(str).str.lower().str.strip()
    return out


def _parse_hours(value):
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            return hours + (minutes / 60)
        except Exception:
            return 0
    try:
        return float(text)
    except Exception:
        return 0


def _coerce_input_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either:
    - attendance.csv schema (date, employee_id, category, status, shift, hours_worked)
    - master_attendance.csv schema (Emp code, At Date, In Time, Late In Flag, Tot Work Time, Emp Group/Contractor)
    """
    expected = {"date", "employee_id", "category", "status"}
    if expected.issubset(set(df.columns)):
        return df

    # Try to map from master_attendance.csv style
    if {"Emp code", "At Date"}.issubset(set(df.columns)):
        out = pd.DataFrame()
        out["employee_id"] = df["Emp code"]
        out["date"] = pd.to_datetime(df["At Date"], dayfirst=True, errors="coerce")
        out["shift"] = df["shift"] if "shift" in df.columns else None

        # Category mapping
        emp_group = df.get("Emp Group")
        contractor_col = df.get("Contractor")
        overall = df.get("Overall")

        def _cat(row):
            eg = str(row.get("Emp Group", "")).strip().lower()
            ov = str(row.get("Overall", "")).strip().lower()
            contractor = str(row.get("Contractor", "")).strip().lower()
            if eg == "contractor":
                return "contractor"
            if contractor == "apprentice" or ov == "apprentice":
                return "app"
            return "onroll"

        out["category"] = df.apply(_cat, axis=1)

        # Status mapping
        in_time = df.get("In Time")
        late_flag = df.get("Late In Flag")

        def _status(row):
            it = row.get("In Time")
            lf = row.get("Late In Flag")
            if pd.isna(it) or str(it).strip() == "":
                return "absent"
            if str(lf).strip() == "1":
                return "late"
            return "present"

        out["status"] = df.apply(_status, axis=1)

        # Hours worked
        if "Tot Work Time" in df.columns:
            out["hours_worked"] = df["Tot Work Time"].apply(_parse_hours)
        else:
            out["hours_worked"] = 0

        return out

    return df


def _compute_daily_rows(df: pd.DataFrame, approved_strength: int, standard_hours: int) -> pd.DataFrame:
    # Build a daily summary table for all dates so we can compute cumulative mandays.
    if df.empty:
        return pd.DataFrame()

    df = _normalize_columns(df)
    df = df[df["date"].notna()].copy()

    daily_rows = []
    for day_value, day_df in df.groupby(df["date"].dt.date):
        day_str = day_value.strftime("%Y-%m-%d")
        day_name = day_value.strftime("%A")

        required_strength = int(round(approved_strength * 0.95))

        # Strength by category (unique employees in that category on that day)
        def _strength(cat: str) -> int:
            return day_df[day_df["category"] == cat]["employee_id"].nunique()

        onroll_strength = _strength("onroll")
        contractor_strength = _strength("contractor")
        app_strength = _strength("app")
        ojt_strength = _strength("ojt")

        # Present counts
        present_df = day_df[day_df["status"].isin(["present", "late"])]

        def _present(cat: str) -> int:
            return present_df[present_df["category"] == cat]["employee_id"].nunique()

        onroll_present = _present("onroll")
        contractor_present = _present("contractor")
        app_present = _present("app")
        ojt_present = _present("ojt")

        total_present = onroll_present + contractor_present + app_present

        # Extra hours
        if "hours_worked" in day_df.columns:
            extra_hours = (
                (day_df["hours_worked"].fillna(0) - standard_hours)
                .clip(lower=0)
                .sum()
            )
        else:
            extra_hours = 0

        extra_mandays = extra_hours / standard_hours if standard_hours else 0
        utilized = total_present + extra_mandays
        gap = utilized - required_strength

        shortage_approved = (
            ((approved_strength - utilized) / approved_strength) * 100
            if approved_strength else None
        )
        shortage_required = (
            ((required_strength - utilized) / required_strength) * 100
            if required_strength else None
        )

        # Absenteeism counts
        absent_df = day_df[day_df["status"] == "absent"]

        def _absent(cat: str) -> int:
            return absent_df[absent_df["category"] == cat]["employee_id"].nunique()

        abs_onroll = _absent("onroll")
        abs_contractor = _absent("contractor")
        abs_app = _absent("app")

        rolling_weekoff = day_df[day_df["status"] == "weekoff"]["employee_id"].nunique()
        informed_leave = day_df[day_df["status"] == "leave"]["employee_id"].nunique()

        row = {
            "Date": day_str,
            "Day": day_name,
            "Approved Strength": approved_strength,
            "Required Strength": required_strength,
            "Onroll": onroll_strength,
            "Contractor": contractor_strength,
            "App": app_strength,
            "C.Off": 0,
            "Total (Onroll Strength)": onroll_strength,
            "OJT": ojt_strength,
            "Onroll Present": onroll_present,
            "Contractor Present": contractor_present,
            "App Present": app_present,
            "Total Present": total_present,
            "OJT Present": ojt_present,
            "Extra Hours": _safe_round(extra_hours, 2),
            "Extra Hours Mandays": _safe_round(extra_mandays, 2),
            "Total Manpower Utilized": _safe_round(utilized, 2),
            "Gap": _safe_round(gap, 2),
            "Cumulative Mandays": None,  # set after we build dataframe
            "Manpower Shortage % (Approved)": _safe_round(shortage_approved, 2),
            "Manpower Shortage % (Required)": _safe_round(shortage_required, 2),
            "Absenteeism Onroll": abs_onroll,
            "Absenteeism Contractor": abs_contractor,
            "Absenteeism App": abs_app,
            "Rolling Shift Week Off": rolling_weekoff,
            "Informed Leave": informed_leave,
        }
        daily_rows.append(row)

    daily_df = pd.DataFrame(daily_rows).sort_values("Date")
    # Cumulative Mandays = running total of Total Present (as a proxy for mandays)
    daily_df["Cumulative Mandays"] = (
        pd.to_numeric(daily_df["Total Present"], errors="coerce")
        .fillna(0)
        .cumsum()
        .round(2)
    )
    return daily_df


def generate_daily_summary(attendance_file: str, risk_file: Optional[str], date_override: Optional[str] = None) -> Optional[str]:
    """
    Generate daily summary CSV for the current date.
    Returns output path or None if no data for today.
    """
    if not os.path.exists(attendance_file):
        print("No attendance data for today")
        return None

    df = pd.read_csv(attendance_file)
    if df.empty:
        print("No attendance data for today")
        return None

    df = _coerce_input_schema(df)

    approved_strength = int(os.getenv("APPROVED_STRENGTH", "300"))
    standard_hours = int(os.getenv("STANDARD_HOURS", "8"))

    daily_df = _compute_daily_rows(df, approved_strength, standard_hours)
    if daily_df.empty:
        print("No attendance data for today")
        return None

    target_date = date_override or date.today().strftime("%Y-%m-%d")
    today_row = daily_df[daily_df["Date"] == target_date]
    if today_row.empty:
        print("No attendance data for today")
        return None

    # Optional: read risk file to count high risk (not included in CSV columns)
    if risk_file and os.path.exists(risk_file):
        try:
            _ = pd.read_csv(risk_file)
        except Exception:
            pass

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"daily_summary_{target_date}.csv")

    # Ensure column order
    today_row = today_row.reindex(columns=SUMMARY_COLUMNS)
    today_row.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    generate_daily_summary("data/attendance.csv", "data/risk_output.csv")
