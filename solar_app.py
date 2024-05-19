from joblib import load
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
forecast = pd.read_csv("data/forecast.csv")
sun = pd.read_excel("data/sunrise-sunset.xlsx")
sun.rename(
    columns={
        "datum": "date",
        "Opkomst": "sunrise",
        "Op ware middag": "sun_noon",
        "Ondergang": "sunset",
    },
    inplace=True,
)

# Clean the data
forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
forecast["date"] = forecast["timestamp"].dt.date.astype("datetime64")
forecast["hour"] = forecast["timestamp"].dt.hour

# Merge the data
data = forecast.merge(sun, on="date")
data.drop(columns=["timestamp"], inplace=True)

# Feature Engineering
# Add light feature
data["after_sunrise"] = data["hour"] >= data["sunrise"].apply(lambda x: x.hour)
data["before_sunset"] = data["hour"] <= data["sunset"].apply(lambda x: x.hour)
data["light"] = data["after_sunrise"] & data["before_sunset"]
data.drop(columns=["after_sunrise", "before_sunset"], inplace=True)
data["light"] = data["light"].astype(int)
# Transform sunrise, sun_noon and sunset to minutes
data["sunrise"] = data["sunrise"].apply(lambda time: time.hour * 60 + time.minute)
data["sun_noon"] = data["sun_noon"].apply(lambda time: time.hour * 60 + time.minute)
data["sunset"] = data["sunset"].apply(lambda time: time.hour * 60 + time.minute)
# minutes since sunrise at the current hour
data["minutes_since_sunrise"] = data["hour"] * 60 - data["sunrise"]
# minutes from noon at the current hour
data["minutes_from_sun_noon"] = abs(data["hour"] * 60 - data["sun_noon"])
# minutes until sunset at the current hour
data["minutes_until_sunset"] = data["sunset"] - data["hour"] * 60
# add month and day columns
data["month"] = data["date"].dt.month
data["day_of_year"] = data["date"].dt.dayofyear
# drop date column
data.drop("date", axis=1, inplace=True)

# Load the model
rf = load("models/random_forest.joblib")

# Predict
preds = rf.predict(data).round(2)

for timestamp, pred in zip(forecast["timestamp"], preds):
    print(f"{timestamp}: {pred}kwh")


# Visualize predictions
plt.figure(figsize=(14, 6))
plt.bar(range(len(preds)), preds, tick_label=forecast["hour"].values)
plt.savefig("predictions/predictions.png")
