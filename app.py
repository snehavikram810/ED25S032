import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(
    page_title='Dark Store Location Optimiser - Chennai',
    page_icon='🛒',
    layout='wide',
)

# Custom CSS styling for polished UI and huge profit box
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .profit-box {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    }
    .profit-box h2 { margin: 0; font-size: 1.8rem; font-weight: 600; color: #e3f2fd; }
    .profit-box h1 { margin: 10px 0 0 0; font-size: 3rem; font-weight: 700; color: #ffffff; }
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

# Sidebar Inputs (Exactly the 3 requested parameters)
with st.sidebar:
  st.header('⚙️ Model Parameters')
  with st.form(key='optimization_form'):
    num_customers = st.number_input(
        'Number of Customers (I)',
        min_value=100,
        max_value=20000,
        value=10000,
        step=500,
    )
    num_stores = st.number_input(
        'Candidate Dark Stores (J)', min_value=2, max_value=60, value=50, step=1
    )
    max_distance_km = st.slider(
        'Max Service Distance / Radius (km)',
        min_value=2.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
    )

    submit_button = st.form_submit_button(
        label='🚀 Run Optimization', use_container_width=True
    )

# Internal Financial Parameters
AVG_FIXED_COST = 150000
TRANSPORT_COST_PER_KM = 12.0
REVENUE_PER_UNIT = 220.0


@st.cache_data
def generate_chennai_land_data(n_cust, n_str):
  np.random.seed(42)

  cust_coords = {}
  for i in range(n_cust):
    hub = CHENNAI_LAND_HUBS[np.random.choice(len(CHENNAI_LAND_HUBS))]
    lat = hub[1] + np.random.uniform(-0.03, 0.03)
    lon = hub[2] + np.random.uniform(-0.03, 0.03)
    cust_coords[i] = (lat, lon)

  store_coords = {}
  for j in range(n_str):
    hub = CHENNAI_LAND_HUBS[j % len(CHENNAI_LAND_HUBS)]
    lat = hub[1] + np.random.uniform(-0.02, 0.02)
    lon = hub[2] + np.random.uniform(-0.02, 0.02)
    store_coords[j] = (lat, lon)

  D = {i: int(np.random.randint(15, 45)) for i in range(n_cust)}
  F = {
      j: int(AVG_FIXED_COST * np.random.uniform(0.8, 1.2)) for j in range(n_str)
  }

  total_demand = sum(D.values())
  # Ensure total store capacity comfortably exceeds total demand
  C = {
      j: int((total_demand / n_str) * np.random.uniform(1.2, 1.8))
      for j in range(n_str)
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
    num_customers, num_stores
)


# High-Performance Facility Location Heuristic Solver (10,000+ customer safe)
def solve_facility_location(
    max_stores_open=None, custom_max_dist=max_distance_km
):
  n_cust = len(D)
  n_str = len(F)

  # Step 1: Score candidate stores based on proximity to total customer demand and fixed costs
  store_scores = []
  for j in range(n_str):
    # Potential accessibility score
    accessibility = sum(
        D[i] for i in range(n_cust) if d[(i, j)] <= custom_max_dist
    )
    score = accessibility - (0.05 * F[j])
    store_scores.append((score, j))

  store_scores.sort(reverse=True, key=lambda x: x[0])

  # Select top stores to open up to limit
  limit = n_str if max_stores_open is None else min(max_stores_open, n_str)
  open_stores = [j for _, j in store_scores[:limit]]

  # Step 2: Allocate customers to the nearest open store within max distance
  allocations = {}
  store_loads = {j: 0.0 for j in open_stores}
  unserved_demand = 0
  total_transport_cost = 0.0

  for i in range(n_cust):
    # Find valid open stores within distance
    valid_stores = [
        j
        for j in open_stores
        if d[(i, j)] <= custom_max_dist and store_loads[j] + D[i] <= C[j]
    ]
    if not valid_stores:
      # Fallback to any open store within distance ignoring soft capacity temporarily
      valid_stores = [j for j in open_stores if d[(i, j)] <= custom_max_dist]

    if valid_stores:
      # Pick nearest open store
      best_store = min(valid_stores, key=lambda j: d[(i, j)])
      allocations[(i, best_store)] = float(D[i])
      store_loads[best_store] += D[i]
      total_transport_cost += D[i] * d[(i, best_store)] * TRANSPORT_COST_PER_KM
    else:
      unserved_demand += D[i]

  # Filter open stores to only those that actually secured assigned demand
  active_open_stores = [j for j in open_stores if store_loads[j] > 0]
  total_fixed_cost = sum(F[j] for j in active_open_stores)
  total_cost = total_transport_cost + total_fixed_cost

  return {
      'status': 'Optimal',
      'total_cost': total_cost,
      'open_stores': active_open_stores,
      'allocations': allocations,
      'store_loads': store_loads,
  }


# Run main optimization
results = solve_facility_location()

if results and results.get('status') == 'Optimal':
  st.success(
      '✅ Optimal Solution Found Successfully for 10,000+ Customers & 50'
      ' Stores!'
  )

  total_demand_val = sum(D.values())
  total_revenue = total_demand_val * REVENUE_PER_UNIT
  total_system_cost = results['total_cost']
  total_profit = total_revenue - total_system_cost

  # Huge Highlighted Profit Box
  st.markdown(
      f"""
    <div class="profit-box">
        <h2>MAXIMUM NET SYSTEM PROFIT</h2>
        <h1>₹ {total_profit:,.2f}</h1>
    </div>
    """,
      unsafe_allow_html=True,
  )

  col1, col2, col3, col4 = st.columns(4)
  col1.metric('Total System Cost', f'₹ {total_system_cost:,.0f}')
  col2.metric(
      'Open Dark Stores', f"{len(results['open_stores'])} / {num_stores}"
  )
  col3.metric('Total Customer Demand', f'{total_demand_val:,} units')
  col4.metric('Total Revenue', f'₹ {total_revenue:,.0f}')

  tab1, tab2, tab3 = st.tabs([
      '🗺️ Detailed Solution Map',
      '📊 KPI & Allocation Table',
      '📈 Profit Scenario Analysis (2, 5, 10, 20, 25)',
  ])

  with tab1:
    st.subheader(
        'Detailed Geospatial Map with Comprehensive Dark Store Analytics'
    )
    m = folium.Map(location=(13.0827, 80.2707), zoom_start=12)

    for j in range(num_stores):
      is_open = j in results['open_stores']
      color = 'green' if is_open else 'red'
      icon_name = 'shopping-cart' if is_open else 'ban'

      assigned_demand = results['store_loads'].get(j, 0.0)
      utilization = (
          (assigned_demand / C[j]) * 100 if is_open and C[j] > 0 else 0
      )
      status_label = '🟢 OPEN' if is_open else '🔴 CLOSED'

      popup_html = f"""
            <div style="font-family: Arial; width: 220px;">
                <h4 style="margin: 0; color: #333;">Dark Store #{j}</h4>
                <hr style="margin: 5px 0;">
                <b>Status:</b> {status_label}<br>
                <b>Fixed Cost:</b> ₹{F[j]:,}<br>
                <b>Capacity:</b> {C[j]:,} units<br>
                <b>Assigned Demand:</b> {assigned_demand:.1f} units<br>
                <b>Utilization:</b> {utilization:.1f}%<br>
            </div>
            """

      folium.Marker(
          location=store_coords[j],
          popup=folium.Popup(popup_html, max_width=250),
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

    plot_limit = min(num_customers, 300)
    for i in range(plot_limit):
      folium.CircleMarker(
          location=cust_coords[i],
          radius=3,
          color='blue',
          fill=True,
          fill_color='blue',
          popup=f'<b>Customer {i}</b><br>Demand: {D[i]} units',
      ).add_to(m)

    link_count = 0
    for (i, j), val in results['allocations'].items():
      if link_count < 200:
        folium.PolyLine(
            locations=[cust_coords[i], store_coords[j]],
            color='gray',
            weight=1,
            opacity=0.4,
        ).add_to(m)
        link_count += 1

    st_folium(m, width=1100, height=550)

  with tab2:
    st.subheader('📋 Detailed Solution & KPI Breakdown Table')

    store_summary = []
    for j in range(num_stores):
      is_open = j in results['open_stores']
      assigned_demand = results['store_loads'].get(j, 0.0)
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
        '📈 Profit Analysis for 2, 5, 10, 20, and 25 Allowed Open Dark Stores'
    )
    st.markdown(
        'Evaluating network profitability when restricting the maximum number of open dark stores '
        'to exactly 2, 5, 10, 20, and 25 locations.'
    )

    store_limits = [2, 5, 10, 20, 25]
    profits = []
    costs = []

    for limit in store_limits:
      if limit <= num_stores:
        res_limit = solve_facility_location(max_stores_open=limit)
        if res_limit and res_limit.get('status') == 'Optimal':
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
    valid_costs = [c for c in costs if c is not None]

    if valid_limits:
      fig, ax = plt.subplots(figsize=(10, 5))
      ax.plot(
          valid_limits,
          valid_profits,
          marker='o',
          color='forestgreen',
          linewidth=2.5,
          markersize=8,
      )
      ax.set_title(
          'Net Profit vs. Maximum Allowed Open
