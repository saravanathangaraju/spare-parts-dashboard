import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

# Set page config
st.set_page_config(page_title="Spare Parts Forecast & Optimization", layout="wide")

# Title
st.title("Spare Parts Forecasting and Inventory Optimization")
st.markdown("A tool to forecast demand and recommend optimal inventory decisions for aerospace maintenance.")

# Load data
@st.cache
def load_data():
    df = pd.read_csv('spare_parts_data.csv')
    feat = pd.read_csv('feature_importance.csv')
    return df, feat

data, feature_importance = load_data()

# Sidebar navigation
page = st.sidebar.radio("Select Page", ["Demand Forecast", "Scenario Simulation", "Inventory Optimization"])

# ===================== Demand Forecast Page =====================
if page == "Demand Forecast":
    st.subheader("Actual vs Predicted Demand")

    if st.checkbox("Show Raw Data"):
        st.dataframe(data)

    st.write("### Data Summary")
    st.write(data.describe())

    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=data['Actual'], y=data['Predicted'], ax=ax)
    ax.plot([data['Actual'].min(), data['Actual'].max()],
            [data['Actual'].min(), data['Actual'].max()], 'r--')
    ax.set_xlabel("Actual Demand")
    ax.set_ylabel("Predicted Demand")
    ax.set_title("Actual vs Predicted Spare Part Demand")
    st.pyplot(fig)

    st.subheader("Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.barplot(x='importance', y='feature', data=feature_importance.sort_values(by='importance', ascending=False), ax=ax2)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

# ===================== Scenario Simulation Page =====================
elif page == "Scenario Simulation":
    st.subheader("Simulate Demand Forecast")

    usage = st.slider("Usage Hours Last 30 Days", min_value=300, max_value=500, value=400)
    failures = st.slider("Failure Count Last 6 Months", min_value=0, max_value=6, value=1)
    disruption = st.selectbox("Supply Disruption", ["No", "Yes"])
    criticality = st.slider("Criticality Score", min_value=6, max_value=10, value=8)
    age = st.slider("Machine Age (Years)", min_value=5, max_value=12, value=8)
    cycles = st.slider("Operational Cycles Last 6 Months", min_value=100, max_value=250, value=150)

    disruption_val = 0 if disruption == "No" else 1

    predicted = (
        usage * 0.05 +
        failures * 2 +
        criticality * 1.5 +
        age * 0.3 +
        cycles * 0.02 +
        disruption_val * 5
    )

    st.write("### Simulation Inputs")
    st.write(f"Usage Hours: {usage}")
    st.write(f"Failures: {failures}")
    st.write(f"Supply Disruption: {disruption}")
    st.write(f"Criticality Score: {criticality}")
    st.write(f"Machine Age: {age}")
    st.write(f"Operational Cycles: {cycles}")

    st.subheader("Predicted Demand")
    st.write(f"Expected Spare Part Demand: {round(predicted)} units")

# ===================== Inventory Optimization Page =====================
elif page == "Inventory Optimization":
    st.subheader("Optimal Order Quantity Simulation")

    demand = st.number_input("Forecasted Demand (units)", value=35)
    holding_cost = st.number_input("Holding Cost per Unit ($)", value=2.0)
    unit_cost = st.number_input("Unit Cost ($)", value=50)
    current_inventory = st.number_input("Current Inventory (units)", value=20)
    backorder_cost = st.number_input("Backorder Penalty per Unit ($)", value=10)

    if st.button("Compute Optimal Order Quantity"):
        # Define optimization variable
        q = cp.Variable(integer=True)

        # Ending inventory and backorder calculation
        ending_inventory = current_inventory + q - demand
        backorder = cp.pos(demand - (current_inventory + q))

        # Objective function
        total_cost = holding_cost * cp.pos(ending_inventory) + unit_cost * q + backorder_cost * backorder
        objective = cp.Minimize(total_cost)

        # Constraints
        constraints = [q >= 0]

        # Solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        st.write("### Optimization Results")
        st.write(f"Optimal Order Quantity: {round(q.value)} units")
        st.write(f"Total Cost: ${problem.value:.2f}")

        if ending_inventory.value is not None:
            st.write(f"Expected Ending Inventory: {round(ending_inventory.value)} units")
        if backorder.value is not None:
            st.write(f"Expected Backorder Amount: {round(backorder.value)} units")

    st.markdown("---")
    st.markdown("This optimization helps in deciding how much inventory to order to balance holding costs and stock-out penalties.")

