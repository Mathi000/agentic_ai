import pandas as pd


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw attendance CSV file
    """
    df = pd.read_csv(file_path)
    return df


def clean_attendance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare attendance data
    """

    # Rename columns to standardized names
    df = df.rename(columns={
        "Emp code": "employee_id",
        "At Date": "date",
        "In Time": "checkin_time",
        "Out Time": "checkout_time",
        "Dept Name": "department"
    })

    # Convert date column
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # Remove rows with missing times
    df = df.dropna(subset=["checkin_time", "checkout_time"])

    # Combine date + time
    df["checkin_time"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["checkin_time"].astype(str),
        errors="coerce"
    )

    df["checkout_time"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["checkout_time"].astype(str),
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(subset=["checkin_time", "checkout_time"])

    # Sort values
    df = df.sort_values(by=["employee_id", "date"])

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def generate_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw logs into daily attendance metrics
    """

    daily_df = df.copy()

    # Calculate working hours
    daily_df["working_hours"] = (
        (daily_df["checkout_time"] - daily_df["checkin_time"])
        .dt.total_seconds() / 3600
    )

    # Office start time assumed 9:00 AM
    office_start = pd.to_datetime("09:00:00").time()

    # Calculate late minutes
    daily_df["late_minutes"] = daily_df["checkin_time"].dt.time.apply(
        lambda x: max(
            (
                pd.Timestamp.combine(pd.Timestamp.today(), x)
                - pd.Timestamp.combine(pd.Timestamp.today(), office_start)
            ).total_seconds() / 60,
            0
        )
    )

    return daily_df


def build_feature_pipeline(file_path: str) -> pd.DataFrame:
    """
    Full pipeline: raw → cleaned → daily → feature engineered
    """

    df = load_raw_data(file_path)
    df = clean_attendance_data(df)
    daily_df = generate_daily_metrics(df)

    daily_df = daily_df.sort_values(by=["employee_id", "date"])

    # Convert check-in time to minutes
    daily_df["checkin_minutes"] = (
        daily_df["checkin_time"].dt.hour * 60 +
        daily_df["checkin_time"].dt.minute
    )

    # Check-in variance per employee
    daily_df["checkin_variance"] = (
        daily_df.groupby("employee_id")["checkin_minutes"]
        .transform("var")
        .fillna(0)
    )

    # Overtime flag (>9 hours)
    daily_df["overtime_flag"] = daily_df["working_hours"].apply(
        lambda x: 1 if x > 9 else 0
    )

    # Late flag
    daily_df["late_flag"] = daily_df["late_minutes"].apply(
        lambda x: 1 if x > 0 else 0
    )

    # Late ratio over last 7 days
    daily_df["late_ratio_7d"] = (
        daily_df.groupby("employee_id")["late_flag"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Consecutive work days
    daily_df["consecutive_work_days"] = (
        daily_df.groupby("employee_id").cumcount() + 1
    )

    # Rolling attendance count (30 days)
    daily_df["days_present_30d"] = (
        daily_df.groupby("employee_id")["date"]
        .rolling(window=30, min_periods=1)
        .count()
        .reset_index(level=0, drop=True)
    )

    # Absence ratio
    daily_df["absence_ratio_30d"] = 1 - (
        daily_df["days_present_30d"] / 30
    )

    # Save processed dataset
    daily_df.to_csv("data/processed/feature_engineered.csv", index=False)

    return daily_df


if __name__ == "__main__":
    final_df = build_feature_pipeline("data/raw/master_attendance.csv")
    print(final_df.head())