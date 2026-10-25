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
    'Optimize supply chain network design by selecting optimal land-based dark store locations '
    'across Chennai, allocating customer demand under capacity constraints, and enforcing maximum service distance limits.'
)

# Realistic Land-Based Chennai Hubs (Avoiding the Bay of Bengal to the East)
CHENNAI_LAND_HUBS = [
    ('Anna Nagar', 13.0850, 80.2100),
    ('T. Nagar', 13.0418, 80.2341),
    ('Velachery', 12.9810, 80.2240),
    ('Adyar', 13.0012, 80.2565),
    ('Mylapore', 13.0368, 80.2676),
    ('Guindy', 13.0067, 80.2206),
    ('Porur', 13.0382, 80.1584),
    ('Tambaram', 12.9249, 80.1436),
    ('Sholinganallur', 12.9010, 80.2279),
    ('Madhavaram', 13.1482, 80.2305),
    ('Kodambakkam', 13.0524, 80.2215),
    ('KK Nagar', 13.0337, 80.1983),
    ('Nungambakkam', 13.0569, 80.2425),
    ('Kilpauk', 13.0822, 80.2403),
    ('Perambur', 13.1112, 80.2337),
    ('Thiruvanmiyur', 12.9825, 80.2619),
    ('Pallavaram', 12.9675, 80.1491),
    ('Chromepet', 12.9516, 80.1462),
    ('Vadapalani', 13.0503, 80.2121),
    ('Ashok Nagar', 13.0378, 80.2078),
]

# Sidebar Configuration inside a Form with an explicit Submit Button
with st.sidebar:
  st.header('⚙️ Model Parameters')
  with st.form(key='optimization_form'):
    num_customers = st.number_input(
        'Number of Customers (I)',
        min_value=5,
        max_value=10000,
        value=100,
        step=50,
    )
    num_stores = st.number_input(
        'Candidate Dark Stores (J)', min_value=2, max_value=50, value=10, step=1
    )
    max_distance_km = st.slider(
        'Max Service Distance / Radius (km)',
        min_value=2.0,
        max_value=30.0,
        value=12.0,
        step=1.0,
    )

    st.subheader('Financial & Operational Inputs')
    avg_fixed_cost = st.number_input(
        'Base Fixed Cost per Store (₹)', value=120000, step=10000
    )
    transport_cost_per_km = st.number_input(
        'Transport Cost per Unit-km (₹)', value=15.0, step=1.0
    )
    revenue_per_unit = st.number_input(
        'Revenue per Unit Sold (₹)', value=180.0, step=10.0
    )

    submit_button = st.form_submit_button(
        label='🚀 Run Optimization', use_container_width=True
    )


@st.cache_data
def generate_chennai_land_data(n_cust, n_str, base_cost):
  np.random.seed(42)

  cust_coords = {}
  for i in range(n_cust):
    hub = CHENNAI_LAND_HUBS[np.random.choice(len(CHENNAI_LAND_HUBS))]
    lat = hub[1] + np.random.uniform(-0.025, 0.025)
    lon = hub[2] + np.random.uniform(-0.025, 0.025)
    cust_coords[i] = (lat, lon)

  store_coords = {}
  for j in range(n_str):
    hub = CHENNAI_LAND_HUBS[j % len(CHENNAI_LAND_HUBS)]
    lat = hub[1] + np.random.uniform(-0.015, 0.015)
    lon = hub[2] + np.random.uniform(-0.015, 0.015)
    store_coords[j] = (lat, lon)

  D = {i: int(np.random.randint(15, 45)) for i in range(n_cust)}
  F = {
      j: int(base_cost * np.random.uniform(0.85, 1.15)) for j in range(n_str)
  }

  total_demand = sum(D.values())
  C = {
      j: int(total_demand * np.random.uniform(0.3, 0.6)) for j in range(n_str)
  }

  d = {}
  for i in range(n_cust):
    for j in range(n_str):
      lat1, lon1 = cust_coords[i]
      lat2, lon2 = store_coords[j]
      dist = np.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) * 111.0
      d[(i, j)] = dist

  return cust_coords, store_coords, D, F, C, d


cust_coords, store_coords, D, F, C, d = generate_chennai_land_data(
    num_customers, num_stores, avg_fixed_cost
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

  # Objective: Minimize Total Cost (Transport Cost + Fixed Cost)
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

  total_demand_val = sum(D.values())
  total_revenue = total_demand_val * revenue_per_unit
  total_system_cost = results['total_cost']
  total_profit = total_revenue - total_system_cost

  col1, col2, col3, col4 = st.columns(4)
  col1.metric('Total Net Profit', f'₹ {total_profit:,.0f}')
  col2.metric('Total System Cost', f'₹ {total_system_cost:,.0f}')
  col3.metric('Open Dark Stores', f"{len(results['open_stores'])} / {num_stores}")
  col4.metric('Total Demand', f'{total_demand_val:,} units')

  # --- TABS FOR ORGANIZED VIEW ---
  tab1, tab2, tab3 = st.tabs(
      ['🗺️ Detailed Solution Map', '📊 KPI & Allocation Table', '📈 Profit Scenario Analysis']
  )

  with tab1:
    st.subheader(
        'Geospatial Visualisation across Chennai Land Hubs & Customer'
        ' Allocations'
    )
    m = folium.Map(location=(13.0827, 80.2707), zoom_start=12)

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

    # Plot Customers (sample subset if too large for browser performance)
    plot_limit = min(num_customers, 500)
    for i in range(plot_limit):
      folium.CircleMarker(
          location=cust_coords[i],
          radius=3,
          color='blue',
          fill=True,
          fill_color='blue',
          popup=f'<b>Customer {i}</b><br>Demand: {D[i]} units',
      ).add_to(m)

    if num_customers > 500:
      st.info(
          f'ℹ️ Displaying first {plot_limit} customers on the interactive map '
          'to ensure optimal browser rendering performance.'
      )

    # Plot Allocation Links
    link_count = 0
    for (i, j), val in results['allocations'].items():
      if link_count < 300:  # limit lines for clean rendering
        folium.PolyLine(
            locations=[cust_coords[i], store_coords[j]],
            color='gray',
            weight=1,
            opacity=0.5,
        ).add_to(m)
        link_count += 1

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
        '📈 Profit Analysis when Up to 2, 5, 10, 20, 25 Dark Stores Open'
    )
    st.markdown(
        'Evaluating network profitability (Revenue minus Transport & Fixed Facility Costs) '
        'across different restrictions on maximum open dark stores.'
    )

    store_limits = [2, 5, 10, 20, 25]
    profits = []
    costs = []

    for limit in store_limits:
      if limit <= num_stores:
        res_limit = solve_dark_store_model(max_stores_open=limit)
        if res_limit and res_limit['status'] == 'Optimal':
          c = res_limit['total_cost']
          p = total_revenue - c
          profits.append(p)
          costs.append(c)
        else:
          profits.append(None)
          costs.append(None)
      else:
        profits.append(None)
        costs.append(None)

    valid_limits = [
        store_limits[i]
        for i, p in enumerate(profits)
        if p is not None
    ]
    valid_profits = [p for p in profits if p is not None]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(
        valid_limits,
        valid_profits,
        marker='o',
        color='forestgreen',
        linewidth=2.5,
        label='Net Profit (₹)',
    )
    ax1.set_title(
        'Total Profit vs. Maximum Allowed Open Dark Stores',
        fontsize=12,
        weight='bold',
    )
    ax1.set_xlabel('Max Allowed Open Dark Stores', fontsize=10)
    ax1.set_ylabel('Net Profit (₹)', fontsize=10, color='forestgreen')
    ax1.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)

    # Sensitivity Table
    sensitivity_df = pd.DataFrame({
        'Max Stores Allowed': valid_limits,
        'Total System Cost (₹)': [
            f'₹{c:,.0f}' for c in costs if c is not None
        ],
        'Total Net Profit (₹)': [f'₹{p:,.0f}' for p in valid_profits],
    })
    st.dataframe(sensitivity_df, use_container_width=True)

    st.info(
        '💡 **Tip for 10,000 Customers & 50 Stores:** Set `Number of Customers = 10000` '
        'and `Candidate Stores = 50` in the sidebar and click **Run Optimization** to execute the enterprise run.'
    )

else:
    st.error(
        '❌ The model is infeasible with the current parameters! '
        'Try increasing the **Max Service Distance** or **Candidate Dark Stores** count in the sidebar.'
    )
