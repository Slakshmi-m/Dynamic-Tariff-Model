import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time

# Settings
sns.set_theme(style="whitegrid")
# plt.rcParams["figure.figsize"] = (12, 6)

print("Libraries imported successfully.")

def fetch_awattar_data():
    # Fetching historical data
    # Note: Using 6 years to cover full history if needed, as requested by user activity
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)
    # Going back 4 years guarantees full coverage of 2022. 
    start_ts = int((pd.Timestamp.now() - pd.DateOffset(years=6)).timestamp() * 1000)
    
    all_data = []
    current = start_ts
    # chunk_size_ms = 60 * 24 * 3600 * 1000 # 60 days chunks to reduce request count
    chunk_size_ms = 100 * 24 * 3600 * 1000 # 100 days chunks based on successful test
    
    print(f"Fetching data from {pd.Timestamp(start_ts, unit='ms')} to {pd.Timestamp(end_ts, unit='ms')}...")
    
    while current < end_ts:
        next_step = min(current + chunk_size_ms, end_ts)
        url = f"https://api.awattar.de/v1/marketdata?start={current}&end={next_step}"
        
        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json().get('data', [])
                    all_data.extend(data)
                    break # Success, move to next chunk
                elif response.status_code == 429:
                    print(f"Rate limited (429) on chunk {current}. Waiting 5s...")
                    time.sleep(5) # Backoff significantly
                else:
                    print(f"Error fetching chunk {current}: Status {response.status_code}")
                    break # Non-retriable error
            except Exception as e:
                print(f"Exception fetching chunk {current}: {e}")
                time.sleep(1)
        
        current = next_step
        # Standard polite sleep between successful requests
        time.sleep(1.0) 
        
    if not all_data:
        raise ValueError("No data returned from Awattar API")
        
    df = pd.DataFrame(all_data)
    # Convert timestamp (ms) to datetime
    df['start_time'] = pd.to_datetime(df['start_timestamp'], unit='ms')
    df['end_time'] = pd.to_datetime(df['end_timestamp'], unit='ms')
    
    # Market price is usually in Eur/MWh, convert to Eur/kWh
    # 1 Eur/MWh = 0.001 Eur/kWh
    df['market_price_eur_kwh'] = df['marketprice'] / 1000.0
    
    df = df.set_index('start_time').sort_index()
    # Remove duplicates just in case
    df = df[~df.index.duplicated(keep='first')]
    return df[['market_price_eur_kwh']]

df_market = fetch_awattar_data()

def generate_synthetic_load(index, total_daily_kwh=10):
    hour_of_day = index.hour
    weights = np.array([
        0.2, 0.2, 0.2, 0.2, 0.3, 0.5,
        1.0, 1.2, 1.0, 0.8, 0.7, 0.6,
        0.6, 0.6, 0.7, 0.9, 1.5, 2.0,
        2.2, 2.0, 1.8, 1.2, 0.6, 0.3
    ])
    load_weights = np.array([weights[h] for h in hour_of_day])
    scaling_factor = total_daily_kwh / weights.sum()
    simulated_load = load_weights * scaling_factor
    noise = np.random.normal(0, 0.05, size=len(simulated_load))
    simulated_load = np.maximum(simulated_load + noise, 0)
    return simulated_load

df_market['load_kwh'] = generate_synthetic_load(df_market.index)

# Parameters
STATIC_RATE = 0.35
FIXED_MARGIN = 0.15 
VENDOR_PROFIT_PORTION = 0.02
CAP_PRICE = 0.45
SMOOTHING_WINDOW = 3

# Tariff Models
df_market['price_static'] = STATIC_RATE
df_market['price_passthrough'] = df_market['market_price_eur_kwh'] + FIXED_MARGIN
df_market['price_capped'] = df_market['price_passthrough'].clip(upper=CAP_PRICE)
df_market['price_smoothed'] = (
    df_market['market_price_eur_kwh'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean() 
    + FIXED_MARGIN
)

# Simulation
GRID_FEES = FIXED_MARGIN - VENDOR_PROFIT_PORTION
results = []
models = ['static', 'passthrough', 'capped', 'smoothed']

for m in models:
    col_name = f'price_{m}'
    total_cost = (df_market['load_kwh'] * df_market[col_name]).sum()
    volatility = df_market[col_name].std()
    hourly_profit = (df_market[col_name] - df_market['market_price_eur_kwh'] - GRID_FEES) * df_market['load_kwh']
    total_profit = hourly_profit.sum()
    
    results.append({
        'Model': m.capitalize(),
        'Total Cost (€)': round(total_cost, 2),
        'Volatility (Std)': round(volatility, 4),
        'Vendor Profit (€)': round(total_profit, 2)
    })

df_results = pd.DataFrame(results)

# Selection
best_model_row = df_results.sort_values('Total Cost (€)').iloc[0]
best_model_name = best_model_row['Model']
best_model_col = f"price_{best_model_name.lower()}"
print(f"Selected Best Model: {best_model_name} (Lowest Cost)")

# --- Business Optimization ---
GRID_AND_TAX_COST = 0.13 
margins_to_test = np.arange(0.02, 0.10, 0.005) 
optimization_results = []
static_total_cost = df_results[df_results['Model'] == 'Static']['Total Cost (€)'].values[0]

for m_margin in margins_to_test:
    price_scenario = df_market['market_price_eur_kwh'] + GRID_AND_TAX_COST + m_margin
    scen_cost = (df_market['load_kwh'] * price_scenario).sum()
    scen_savings_pct = (static_total_cost - scen_cost) / static_total_cost * 100
    scen_profit = m_margin * df_market['load_kwh'].sum()
    
    optimization_results.append({
        'Margin (€/kWh)': round(m_margin, 3),
        'Customer Cost (€)': round(scen_cost, 2),
        'Savings vs Static (%)': round(scen_savings_pct, 2),
        'Vendor Profit (€)': round(scen_profit, 2)
    })

df_opt = pd.DataFrame(optimization_results)
sweet_spots = df_opt[df_opt['Savings vs Static (%)'] > 20]

if not sweet_spots.empty:
    best_balanced = sweet_spots.sort_values('Vendor Profit (€)', ascending=False).iloc[0]
    print("\nBalanced Business Model Found:")
    print(best_balanced)
else:
    print("No balanced model found.")

# ML
df_ml = df_market.copy()
df_ml['hour'] = df_ml.index.hour
df_ml['day_of_week'] = df_ml.index.dayofweek
df_ml['lag_1h'] = df_ml[best_model_col].shift(1)
df_ml = df_ml.dropna()

if len(df_ml) > 100:
    features = ['hour', 'day_of_week', 'lag_1h']
    target = best_model_col
    X = df_ml[features]
    y = df_ml[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    print(f"\nRandom Forest RMSE: {rmse_rf:.4f}")
else:
    print("Not enough data for ML step.")
