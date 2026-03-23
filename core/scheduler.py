import os
import time

from dotenv import load_dotenv

from daily_summary import generate_daily_summary
from email_service import send_email, load_email_env


def run_daily_job():
    """
    Pipeline:
    1) Generate daily summary
    2) Email the summary (retry once on failure)
    """
    load_dotenv()

    attendance_file = os.getenv("ATTENDANCE_FILE", "data/attendance.csv")
    risk_file = os.getenv("RISK_FILE", "data/risk_output.csv")

    date_override = os.getenv("DATE_OVERRIDE")
    output_path = generate_daily_summary(attendance_file, risk_file, date_override)
    if not output_path:
        print("No attendance data for today")
        return

    sender, app_password, receiver = load_email_env()
    if not sender or not app_password or not receiver:
        print("Email credentials not configured")
        return

    ok = send_email(sender, app_password, receiver, output_path)
    if not ok:
        time.sleep(2)
        ok = send_email(sender, app_password, receiver, output_path)

    if not ok:
        print("Email failed after retry")


if __name__ == "__main__":
    run_daily_job()
