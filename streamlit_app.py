import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cvxpy as cp

# Page configuration
st.set_page_config(page_title="Spare Parts Forecast & Optimization", layout="wide")

# Title
st.title("Spare Parts Forecasting and Inventory Optimization")
st.markdown("A decision support tool for aerospace spare parts planning using predictive analytics and optimization.")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('spare_parts_data.csv')
    except Exception:
        df = pd.DataFrame({
            "Actual": np.random.poisson(20, 100),
            "Predicted": np.random.poisson(18, 100)
        })
    try:
        feat = pd.read_csv('feature_importance.csv')
    except Exception:
        feat = pd.DataFrame({
            "feature": ["usage", "failures", "criticality", "age", "cycles", "disruption"],
            "importance": [0.3, 0.2, 0.18, 0.12, 0.12, 0.08]
        })
    return df, feat

data, feature_importance = load_data()

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Demand Forecast", "Scenario Simulation", "Inventory Optimization"])

# ===================== Demand Forecast =====================
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
    ax.set_title("Actual vs Predicted Demand")
    st.pyplot(fig)

    st.subheader("Feature Importance")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.barplot(x='importance', y='feature', data=feature_importance.sort_values(by='importance', ascending=False), ax=ax2)
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

# ===================== Scenario Simulation =====================
elif page == "Scenario Simulation":
    st.subheader("Simulate Spare Part Demand")

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

# ===================== Inventory Optimization =====================
elif page == "Inventory Optimization":
    st.subheader("Inventory Optimization using MILP")

    demand = st.number_input("Forecasted Demand (units)", value=35, min_value=0)
    holding_cost = st.number_input("Holding Cost per Unit ($)", value=2.0, min_value=0.0, format="%.2f")
    unit_cost = st.number_input("Unit Cost ($)", value=50.0, min_value=0.0, format="%.2f")
    current_inventory = st.number_input("Current Inventory (units)", value=20, min_value=0)
    backorder_cost = st.number_input("Backorder Penalty per Unit ($)", value=10.0, min_value=0.0, format="%.2f")
    max_order = st.number_input("Max Order Limit (units, optional)", value=100, min_value=1)

    if st.button("Compute Optimal Order Quantity"):
        # Define optimization variable (integer non-negative)
        q = cp.Variable(integer=True)

        # Ending inventory and backorder calculation (expressions)
        ending_inventory = current_inventory + q - demand
        backorder = cp.pos(demand - (current_inventory + q))

        # Objective function
        total_cost = holding_cost * cp.pos(ending_inventory) + unit_cost * q + backorder_cost * backorder
        objective = cp.Minimize(total_cost)

        # Constraints
        constraints = [
            q >= 0,
            q <= int(max_order)  # practical upper bound to aid solver
        ]

        # Solve problem (try a mixed-integer solver if available)
        problem = cp.Problem(objective, constraints)
        try:
            # prefer GLPK_MI if available for integer problems
            problem.solve(solver=cp.GLPK_MI)
        except Exception:
            # fallback to default solver
            problem.solve()

        st.write("### Optimization Results")

        # Check solver status
        status = problem.status
        if status in [cp.OPTIMAL, "optimal", "Optimal"]:
            # Safely extract scalar values (convert cvxpy/np types to python floats)
            def safe_scalar(expr):
                val = expr.value
                if val is None:
                    return None
                # squeeze and convert to python float
                return float(np.squeeze(val))

            q_val = safe_scalar(q)
            ending_val = safe_scalar(ending_inventory)
            backorder_val = safe_scalar(backorder)

            if q_val is not None:
                st.write(f"Optimal Order Quantity: {int(round(q_val))} units")
            else:
                st.write("Could not compute optimal order quantity.")

            if ending_val is not None:
                st.write(f"Expected Ending Inventory: {int(round(ending_val))} units")
            else:
                st.write("Could not compute expected ending inventory.")

            if backorder_val is not None:
                st.write(f"Expected Backorder Amount: {int(round(backorder_val))} units")
            else:
                st.write("Could not compute expected backorder amount.")

            st.write(f"Total cost (objective): ${problem.value:.2f}")
        else:
            st.write(f"Solver returned status: {status}. No valid solution found.")

    st.markdown("---")
    st.markdown("This module helps you determine the right inventory levels balancing cost and availability.")