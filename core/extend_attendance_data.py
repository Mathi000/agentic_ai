import pandas as pd

df = pd.read_csv("data/raw/master_attendance.csv")

df.columns = df.columns.str.strip()

df["At Date"] = pd.to_datetime(df["At Date"], dayfirst=True, errors="coerce")

all_months = []

for i in range(6):
    temp = df.copy()
    temp["At Date"] = temp["At Date"] + pd.DateOffset(months=i)
    all_months.append(temp)

extended_df = pd.concat(all_months, ignore_index=True)

extended_df.to_csv("data/raw/master_attendance.csv", index=False)

print("Extended dataset created:", len(extended_df))