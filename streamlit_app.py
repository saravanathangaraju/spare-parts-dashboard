import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------------
# Header Section
# -------------------------------
st.set_page_config(page_title="Spare Parts Inventory Dashboard", layout="wide")
st.title("🛠️ Spare Parts Forecasting & Optimization Dashboard")

# -------------------------------
# Load Simulated Dataset
# -------------------------------
@st.cache_data
def load_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'machine_age': np.random.randint(1, 15, 50),
        'operational_cycles': np.random.randint(200, 1000, 50),
        'external_event_index': np.random.normal(0, 1, 50),
        'part_failure_rate': np.random.uniform(0.01, 0.2, 50),
        'lead_time_days': np.random.randint(3, 20, 50),
    })
    data['spare_part_demand'] = (
        50
        + 2 * data['machine_age']
        + 0.01 * data['operational_cycles']
        + 20 * data['part_failure_rate']
        + 5 * data['external_event_index']
        + np.random.normal(0, 5, 50)
    ).round().astype(int)
    return data


df = load_data()

# -------------------------------
# Sidebar Navigation
# -------------------------------
section = st.sidebar.radio(
    "Select Section",
    [
        "📊 Exploratory Data",
        "📈 Forecasting (XGBoost)",
        "📦 Inventory Optimization (MILP)",
    ],
)

# -------------------------------
# 📊 Exploratory Data Section
# -------------------------------
if section == "📊 Exploratory Data":
    st.subheader("Exploratory Data Overview")
    st.dataframe(df)

    st.markdown("#### 📌 Demand Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['spare_part_demand'], bins=15, color='skyblue', edgecolor='black')
    ax.set_title("Spare Part Demand Distribution")
    ax.set_xlabel("Demand")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# -------------------------------
# 📈 Forecasting with XGBoost
# -------------------------------
elif section == "📈 Forecasting (XGBoost)":
    st.subheader("XGBoost Forecasting Model")

    features = [
        'machine_age',
        'operational_cycles',
        'external_event_index',
        'part_failure_rate',
        'lead_time_days',
    ]
    target = 'spare_part_demand'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.markdown("##### 🔍 Model Evaluation")
    st.write("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
    st.write("RMSE:", round(mean_squared_error(y_test, y_pred, squared=False), 2))

    st.markdown("##### 📉 Actual vs Predicted")
    comparison_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred.round()
    }).reset_index(drop=True)

    st.dataframe(comparison_df)

    fig, ax = plt.subplots()
    ax.plot(comparison_df['Actual'], label='Actual', marker='o')
    ax.plot(comparison_df['Predicted'], label='Predicted', marker='x')
    ax.set_title("Predicted vs Actual Demand")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Demand")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# 📦 Inventory Optimization (MILP)
# -------------------------------
elif section == "📦 Inventory Optimization (MILP)":
    st.subheader("MILP-Based Inventory Optimization")

    # Input parameters
    annual_demand = st.number_input("Annual Demand (D)", min_value=100, value=1000)
    ordering_cost = st.number_input("Ordering Cost (S)", min_value=1, value=500)
    holding_cost = st.number_input("Holding Cost per Unit (H)", min_value=1, value=10)
    backorder_cost = st.number_input("Backorder Cost (B)", min_value=1, value=50)
    safety_stock = st.number_input("Safety Stock (SS)", min_value=0, value=100)

    if st.button("Compute Optimal Order Quantity"):
        # Define integer decision variable
        q = cp.Variable(integer=True, name="order_quantity")

        try:
            # Define the cost expression using cvxpy expressions
            total_cost = (annual_demand / q) * ordering_cost + (q / 2) * holding_cost + safety_stock * backorder_cost
            objective = cp.Minimize(total_cost)
            constraints = [q >= 1, q <= 5000]

            problem = cp.Problem(objective, constraints)
            result = problem.solve(solver=cp.SCS)

            if problem.status in ["optimal", "optimal_inaccurate"]:
                st.success("✅ Optimization completed successfully.")
                # q.value may be a float; round for display
                optimal_q = int(round(float(q.value)))
                # total_cost.value might be None if solver failed to assign; guard it
                total_cost_val = float(total_cost.value) if total_cost.value is not None else float(result)
                st.write(f"🔢 Optimal Order Quantity: **{optimal_q} units**")
                st.caption(f"Total Estimated Cost: ₹{round(total_cost_val):,}")
            else:
                st.warning(f"⚠️ Optimization did not converge. Status: `{problem.status}`")
        except Exception as e:
            st.error(f"❌ Optimization failed due to: `{str(e)}`")