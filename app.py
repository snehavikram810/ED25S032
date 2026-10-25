import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
import streamlit as st
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(
    page_title='Dark Store Location Optimiser - Chennai',
    page_icon='🛒',
    layout='wide',
)

# Custom CSS styling for a polished UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
""", unsafe_allow_html=True)

st.title('🛒 Dark Store Location Optimisation System (Chennai Region)')
st.markdown(
    'Optimize supply chain network design by selecting optimal dark store locations, '
    'allocating customer demand under capacity constraints, and enforcing maximum service distance limits.'
)

# Sidebar Configuration Inputs
st.sidebar.header('⚙️ Model Parameters')

num_customers = st.sidebar.number_input(
    'Number of Customers (I)', min_value=5, max_value=2000, value=50, step=5
)
num_stores = st.sidebar.number_input(
    'Candidate Dark Stores (J)', min_value=2, max_value=100, value=10, step=1
)
max_distance_km = st.sidebar.slider(
    'Max Service Distance / Radius (km)',
    min_value=2.0,
    max_value=50.0,
    value=15.0,
    step=1.0,
)

st.sidebar.subheader('Financial & Operational Inputs')
avg_fixed_cost = st.sidebar.number_input(
    'Base Fixed Cost per Store (₹)', value=120000, step=10000
)
transport_cost_per_km = st.sidebar.number_input(
    'Transport Cost per Unit-km (₹)', value=15.0, step=1.0
)

# Chennai Center Coordinates
CHENNAI_CENTER = (13.0827, 80.2707)


@st.cache_data
def generate_chennai_data(n_cust, n_str):
  np.random.seed(42)
  # Generate customer locations around Chennai (approx ±0.15 degrees ~ 15-20 km)
  cust_coords = {}
  for i in range(n_cust):
    lat = CHENNAI_CENTER[0] + np.random.uniform(-0.12, 0.12)
    lon = CHENNAI_CENTER[1] + np.random.uniform(-0.12, 0.12)
    cust_coords[i] = (lat, lon)

  store_coords = {}
  for j in range(n_str):
    lat = CHENNAI_CENTER[0] + np.random.uniform(-0.10, 0.10)
    lon = CHENNAI_CENTER[1] + np.random.uniform(-0.10, 0.10)
    store_coords[j] = (lat, lon)

  np.random.seed(42)
  D = {i: int(np.random.randint(15, 45)) for i in range(n_cust)}
  F = {
      j: int(avg_fixed_cost * np.random.uniform(0.8, 1.2))
      for j in range(n_str)
  }

  total_demand = sum(D.values())
  C = {
      j: int(total_demand * np.random.uniform(0.3, 0.6)) for j in range(n_str)
  }

  # Haversine distance in km approximation
  d = {}
  for i in range(n_cust):
    for j in range(n_str):
      lat1, lon1 = cust_coords[i]
      lat2, lon2 = store_coords[j]
      # Approximate km distance (~111 km per degree)
      dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111.0
      d[(i, j)] = dist

  return cust_coords, store_coords, D, F, C, d


cust_coords, store_coords, D, F, C, d = generate_chennai_data(
    num_customers, num_stores
)


# Optimization Execution Function
def solve_dark_store_model(
    max_stores_open=None, custom_max_dist=max_distance_km
):
  solver = pywraplp.Solver.CreateSolver('CBC')
  if not solver:
    return None

  X = {}
  for i in range(num_customers):
    for j in range(num_stores):
      X[(i, j)] = solver.NumVar(0.0, solver.infinity(), f'X_{i}_{j}')

  y = {}
  for j in range(num_stores):
    y[j] = solver.IntVar(0, 1, f'y_{j}')

  # Objective: Minimize Transport Cost + Fixed Cost
  objective = solver.Objective()
  for i in range(num_customers):
    for j in range(num_stores):
      objective.SetCoefficient(X[(i, j)], d[(i, j)] * transport_cost_per_km)
  for j in range(num_stores):
    objective.SetCoefficient(y[j], F[j])
  objective.SetMinimization()

  # Constraint 1: Demand Satisfaction
  for i in range(num_customers):
    constraint = solver.Constraint(D[i], D[i], f'Demand_{i}')
    for j in range(num_stores):
      constraint.SetCoefficient(X[(i, j)], 1.0)

  # Constraint 2: Facility Capacity
  for j in range(num_stores):
    constraint = solver.Constraint(-solver.infinity(), 0.0, f'Capacity_{j}')
    for i in range(num_customers):
      constraint.SetCoefficient(X[(i, j)], 1.0)
    constraint.SetCoefficient(y[j], -C[j])

  # Constraint 3: Maximum Distance / Coverage limit per assignment
  for i in range(num_customers):
    for j in range(num_stores):
      if d[(i, j)] > custom_max_dist:
        solver.Add(X[(i, j)] <= 0)

  # Optional limit on max stores open
  if max_stores_open is not None:
    store_count_constraint = solver.Constraint(
        0, max_stores_open, 'Max_Stores_Open_Limit'
    )
    for j in range(num_stores):
      store_count_constraint.SetCoefficient(y[j], 1.0)

  status = solver.Solve()

  if status == pywraplp.Solver.OPTIMAL:
    total_cost = solver.Objective().Value()
    open_stores = [j for j in range(num_stores) if y[j].solution_value() > 0.5]
    allocations = {
        (i, j): X[(i, j)].solution_value()
        for i in range(num_customers)
        for j in range(num_stores)
        if X[(i, j)].solution_value() > 1e-5
    }
    return {
        'status': 'Optimal',
        'total_cost': total_cost,
        'open_stores': open_stores,
        'allocations': allocations,
        'y': y,
        'X': X,
    }
  else:
    return {'status': 'Infeasible'}


# Run main optimization
results = solve_dark_store_model()

if results and results['status'] == 'Optimal':
  st.success('✅ Optimal Solution Found Successfully!')

  # --- KPI METRICS ---
  total_demand_val = sum(D.values())
  total_fixed_cost = sum(F[j] for j in results['open_stores'])
  total_transport_cost = sum(
      results['allocations'][(i, j)] * d[(i, j)] * transport_cost_per_km
      for i, j in results['allocations']
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric('Total System Cost', f"₹ {results['total_cost']:,.0f}")
  col2.metric('Open Dark Stores', f"{len(results['open_stores'])} / {num_stores}")
  col3.metric('Total Customer Demand', f'{total_demand_val} units')
  col4.metric('Avg Delivery Distance', f"{np.mean(list(d.values())):.2f} km")

  # --- TABS FOR ORGANIZED VIEW ---
  tab1, tab2, tab3 = st.tabs(
      ['🗺️ Detailed Solution Map', '📊 KPI & Allocation Table', '📈 Scenario Analysis']
  )

  with tab1:
    st.subheader('Geospatial Visualisation of Dark Stores & Customer Allocations')
    m = folium.Map(location=CHENNAI_CENTER, zoom_start=13)

    # Plot Stores
    for j in range(num_stores):
      is_open = j in results['open_stores']
      color = 'green' if is_open else 'red'
      icon_name = 'shopping-cart' if is_open else 'ban'
      status_text = (
          f'OPEN (Cap: {C[j]}, Cost: ₹{F[j]:,})'
          if is_open
          else 'CLOSED (Candidate)'
      )

      folium.Marker(
          location=store_coords[j],
          popup=f'<b>Dark Store {j}</b><br>Status: {status_text}',
          icon=folium.Icon(color=color, icon=icon_name, prefix='fa'),
      ).add_to(m)

      if is_open:
        folium.Circle(
            location=store_coords[j],
            radius=max_distance_km * 1000,
            color='green',
            weight=1,
            fill=True,
            fill_opacity=0.05,
        ).add_to(m)

    # Plot Customers
    for i in range(num_customers):
      folium.CircleMarker(
          location=cust_coords[i],
          radius=4,
          color='blue',
          fill=True,
          fill_color='blue',
          popup=f'<b>Customer {i}</b><br>Demand: {D[i]} units',
      ).add_to(m)

    # Plot Allocation Links
    for (i, j), val in results['allocations'].items():
      folium.PolyLine(
          locations=[cust_coords[i], store_coords[j]],
          color='gray',
          weight=1.5,
          opacity=0.6,
      ).add_to(m)

    st_folium(m, width=1100, height=550)

  with tab2:
    st.subheader('📋 Detailed Solution & KPI Breakdown')

    store_summary = []
    for j in range(num_stores):
      is_open = j in results['open_stores']
      assigned_demand = sum(
          val for (cust, store), val in results['allocations'].items() if store == j
      )
      utilization = (
          (assigned_demand / C[j]) * 100 if is_open and C[j] > 0 else 0
      )
      store_summary.append({
          'Store ID': f'Store {j}',
          'Status': 'Open 🟢' if is_open else 'Closed 🔴',
          'Fixed Cost (₹)': f'₹{F[j]:,}',
          'Capacity': C[j],
          'Assigned Demand': f'{assigned_demand:.1f}',
          'Capacity Utilization (%)': f'{utilization:.1f}%',
      })

    st.dataframe(pd.DataFrame(store_summary), use_container_width=True)

  with tab3:
    st.subheader(
        '📈 Network Cost vs. Number of Open Dark Stores (Sensitivity Analysis)'
    )
