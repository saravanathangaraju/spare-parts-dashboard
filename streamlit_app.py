import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Spare Parts Forecast Dashboard", layout="wide")

# Title
st.title("Spare Parts Demand Forecast Dashboard")
st.markdown("This dashboard provides insights into spare part demand forecasting for aerospace maintenance operations.")

# Load data
@st.cache
def load_data():
    df = pd.read_csv('spare_parts_data.csv')
    feat = pd.read_csv('feature_importance.csv')
    return df, feat

data, feature_importance = load_data()

# Sidebar filters
st.sidebar.header("Filters")

# Example filter by Part ID
part_ids = data['Actual'].unique()
selected_part = st.sidebar.selectbox("Select Part Index", data.index)

# Show dataset
if st.checkbox("Show Raw Data"):
    st.subheader("Spare Parts Demand Data")
    st.dataframe(data)

# Summary stats
st.subheader("Data Summary")
st.write(data.describe())

# Plot: Actual vs Predicted
st.subheader("Actual vs Predicted Spare Part Demand")
fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=data['Actual'], y=data['Predicted'], ax=ax1)
ax1.plot([data['Actual'].min(), data['Actual'].max()],
         [data['Actual'].min(), data['Actual'].max()], 'r--')
ax1.set_xlabel("Actual Demand")
ax1.set_ylabel("Predicted Demand")
st.pyplot(fig1)

# Feature importance
st.subheader("Feature Importance")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x='importance', y='feature', data=feature_importance, ax=ax2)
ax2.set_title("Features Ranked by Importance")
st.pyplot(fig2)

# Scenario Simulation
st.subheader("Scenario Simulation")

usage = st.slider("Usage Hours Last 30 Days", min_value=300, max_value=500, value=400)
failures = st.slider("Failure Count Last 6 Months", min_value=0, max_value=6, value=1)
supply_disruption = st.selectbox("Supply Disruption", ["No", "Yes"])

disruption_value = 0 if supply_disruption == "No" else 1

st.write("Simulation Inputs")
st.write(f"Usage Hours: {usage}")
st.write(f"Failure Count: {failures}")
st.write(f"Supply Disruption: {supply_disruption}")

# Simulate prediction (dummy formula similar to training logic)
predicted_demand = (
    usage * 0.05 +
    failures * 2 +
    8 * 1.5 +        # Example criticality score
    8 * 0.3 +        # Example machine age
    150 * 0.02 +     # Example operational cycles
    disruption_value * 5 +
    0                # Noise excluded for simulation
)

st.subheader("Simulated Forecast")
st.write(f"Predicted Demand: {round(predicted_demand)} spare parts")

# Footer
st.markdown("---")
st.markdown("Built as part of an MBA Business Analytics project on predictive maintenance and inventory optimization.")

