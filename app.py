import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

st.set_page_config(page_title="IoT Analytics Dashboard", layout="wide")

st.title("🌐 Interactive IoT Sensor Analytics Dashboard")
st.markdown(
    "This dashboard analyzes environmental IoT sensor data and uses a "
    "Random Forest model to predict temperature from humidity and air quality."
)

@st.cache_data
def load_data():
    data_path = Path(__file__).resolve().parent / "data" / "iot_sensor_data.csv"
    data = pd.read_csv(data_path)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"])
    return data

@st.cache_resource
def train_model(data):
    X = data[["Humidity", "Air_Quality"]]
    y = data["Temperature"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)

    return model, mae, rmse

df = load_data()
model, mae, rmse = train_model(df)

st.sidebar.header("🎛️ Control Panel")
display_hours = st.sidebar.slider(
    "Historical Time Window (Hours)",
    min_value=24,
    max_value=168,
    value=72,
)

st.sidebar.subheader("🔮 Predict Target Temperature")
input_humidity = st.sidebar.slider(
    "Current Humidity (%)",
    min_value=10.0,
    max_value=90.0,
    value=50.0,
)
input_aqi = st.sidebar.slider(
    "Current Air Quality (AQI)",
    min_value=0,
    max_value=300,
    value=50,
)

st.subheader("📊 Current Environmental Status")
col1, col2, col3 = st.columns(3)

subset = df.head(display_hours)
latest_temp = subset["Temperature"].iloc[-1]
latest_hum = subset["Humidity"].iloc[-1]
latest_aqi = subset["Air_Quality"].iloc[-1]

col1.metric("Latest Temperature", f"{latest_temp:.2f} °C")
col2.metric("Latest Humidity", f"{latest_hum:.2f} %")
col3.metric("Air Quality Index", f"{latest_aqi} AQI")

st.markdown("---")

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📈 Sensor Historical Trends")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(subset["Timestamp"], subset["Temperature"], label="Temperature (°C)")
    ax.plot(subset["Timestamp"], subset["Humidity"], label="Humidity (%)")
    ax.set_xlabel("Timestamp")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)

with right_col:
    st.subheader("🤖 ML Performance")
    st.info("**Model:** Random Forest Regressor")

    prediction_input = pd.DataFrame(
        [[input_humidity, input_aqi]],
        columns=["Humidity", "Air_Quality"],
    )
    predicted_temp = model.predict(prediction_input)[0]

    st.metric(
        "🔮 Predicted Temperature Output",
        f"{predicted_temp:.2f} °C",
    )

    st.markdown("#### Model Accuracy")
    st.write(f"• **MAE:** {mae:.4f} °C")
    st.write(f"• **RMSE:** {rmse:.4f} °C")
