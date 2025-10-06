import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as plt
st.set_page_config(page_title="ZAMTOLOV:The Traffic volume predictor....", layout="centered")
st.title("🚦 ZAMTOLOV : The traffic volume predictor")
df = pd.read_csv("south_chennai_traffic_2019_20251.csv", parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'],format='%d-%m-%Y %H:%M')
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['day_of_week'] = df['date'].dt.dayofweek
st.subheader(" Average Traffic Volume per Hour")
avg_traffic_by_hour = df.groupby('hour')['traffic_volume'].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(avg_traffic_by_hour.index, avg_traffic_by_hour.values)
ax.set_title('Average Traffic Volume per Hour of Day')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Average Traffic Volume')
ax.set_xticks(range(0, 24))
ax.grid(axis='y')
st.pyplot(fig)
features = ['highway_number', 'day_of_week', 'hour', 'rainfall_mm', 'temperature_c', 'holiday', 'month', 'year']
X = df[features]
y = df['traffic_volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("🔮 Predict Traffic Volume")
date_input = st.text_input("Enter date and time (YYYY-MM-DD HH:MM)", value="2025-06-15 08:00")
highway_number = st.text_input("Select Highway Number",value=35)
holiday = st.radio("Is it a holiday?", ["No", "Yes"]) == "Yes"
rainfall_mm = st.number_input("Rainfall (in mm)", min_value=0.0, value=0.0)

if rainfall_mm > 0:
    temperature_c = 32 - min(rainfall_mm * 0.5, 5)
else:
    temperature_c = 32

if st.button("Predict"):
    try:
        user_datetime = datetime.strptime(date_input, "%Y-%m-%d %H:%M")
        user_input = pd.DataFrame({
            'highway_number': [highway_number],
            'day_of_week': [user_datetime.weekday()],
            'hour': [user_datetime.hour],
            'rainfall_mm': [rainfall_mm],
            'temperature_c': [temperature_c],
            'holiday': [1 if holiday else 0],
            'month': [user_datetime.month],
            'year': [user_datetime.year]
        })

        prediction = model.predict(user_input)[0]
        st.success(f"Predicted traffic volume on {date_input} for highway {highway_number}: **{int(prediction)} vehicles**")
    except Exception as e:
        st.error(f"Error: {e}")

