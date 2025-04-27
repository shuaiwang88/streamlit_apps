import streamlit as st
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

# ================== Simple Login ==================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔒 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state.logged_in = True
            st.success("✅ Logged in successfully!")
        else:
            st.error("❌ Invalid username or password.")
    st.stop()

# ================== Streamlit App ==================
st.title("📦 Facility Location Optimization")
st.markdown("Optimize warehouse/plant locations using Pyomo and HiGHS solver.")

# ================== Possible Extensions (Collapsible Section) ==================
with st.expander("🚀 **Possible Extensions (Click to Expand)**"):
    st.markdown("""
    **Enhance this model with advanced features:**  
    - **Stochastic Demand**: Model uncertainty using `PySP`.  
    - **Multi-Objective**: Minimize cost + carbon emissions.  
    - **GIS Integration**: Plot facilities on a map with `Folium`.  
    - **Dynamic Relocation**: Optimize over time periods.  
    - **Vehicle Routing**: Combine with `OR-Tools` for last-mile delivery.  
    - **API Deployment**: Wrap the solver in `FastAPI` for programmatic access.  
    - **Explainable AI**: Use `SHAP` to interpret facility choices.  
    """)
    st.write("🔍 *Contact us to implement these!*")

# ================== Input Data ==================
st.sidebar.header("Scenario Settings")
num_customers = st.sidebar.slider("Number of Customers", 5, 20, 10)
num_facilities = st.sidebar.slider("Number of Potential Facilities", 3, 10, 5)

# Randomly generate data (same as before)
np.random.seed(42)
customers = [f"C{i+1}" for i in range(num_customers)]
facilities = [f"F{j+1}" for j in range(num_facilities)]
demand = {i: np.random.randint(10, 100) for i in customers}
transport_cost = pd.DataFrame(
    np.random.randint(5, 50, size=(num_customers, num_facilities)),
    index=customers,
    columns=facilities,
)
fixed_cost = {j: np.random.randint(500, 2000) for j in facilities}
capacity = {j: np.random.randint(100, 500) for j in facilities}

# ================== Pyomo Model & Results ==================
def solve_facility_location():
    model = pyo.ConcreteModel()
    model.I = pyo.Set(initialize=customers)
    model.J = pyo.Set(initialize=facilities)
    model.d = pyo.Param(model.I, initialize=demand)
    model.c = pyo.Param(model.I, model.J, initialize=lambda m, i, j: transport_cost.loc[i, j])
    model.f = pyo.Param(model.J, initialize=fixed_cost)
    model.u = pyo.Param(model.J, initialize=capacity)
    model.y = pyo.Var(model.J, domain=pyo.Binary)
    model.x = pyo.Var(model.I, model.J, domain=pyo.NonNegativeReals)
    
    def total_cost_rule(m):
        return sum(m.f[j] * m.y[j] for j in m.J) + sum(m.c[i,j] * m.x[i,j] for i in m.I for j in m.J)
    model.total_cost = pyo.Objective(rule=total_cost_rule, sense=pyo.minimize)
    
    def demand_satisfaction_rule(m, i):
        return sum(m.x[i,j] for j in m.J) == 1
    model.demand_satisfaction = pyo.Constraint(model.I, rule=demand_satisfaction_rule)
    
    def capacity_rule(m, j):
        return sum(m.d[i] * m.x[i,j] for i in m.I) <= m.u[j] * m.y[j]
    model.capacity_constraint = pyo.Constraint(model.J, rule=capacity_rule)
 
    solver = SolverFactory('appsi_highs')
    results = solver.solve(model)
    return model, results

if st.button("Run Optimization"):
    model, results = solve_facility_location()
    st.success("✅ Optimization Solved Successfully!")
    
    # Display results (same as before)
    opened_facilities = [j for j in model.J if pyo.value(model.y[j]) > 0.5]
    st.write(f"**Opened Facilities:** {', '.join(opened_facilities)}")
    st.write(f"**Total Cost:** ${pyo.value(model.total_cost):,.2f}")

    # Plot allocations
    fig, ax = plt.subplots()
    for j in opened_facilities:
        customers_served = [i for i in model.I if pyo.value(model.x[i,j]) > 0]
        ax.scatter(
            [transport_cost.loc[i,j] for i in customers_served],
            [demand[i] for i in customers_served],
            label=f"Facility {j}",
            s=100,
        )
    ax.set_xlabel("Transport Cost")
    ax.set_ylabel("Customer Demand")
    ax.legend()
    st.pyplot(fig)

# ================== Data Preview ==================
st.sidebar.subheader("Generated Data Preview")
if st.sidebar.checkbox("Show Demand Data"):
    st.sidebar.write("**Customer Demand:**", demand)
if st.sidebar.checkbox("Show Transport Costs"):
    st.sidebar.dataframe(transport_cost)