# Project Code Explanation: Dynamic Tariff Optimization

This document provides a detailed breakdown of the code within `dynamic_tariff.ipynb`. The notebook simulates various dynamic electricity tariff models, compares their economic viability for both vendors and customers, and evaluates their price predictability using Machine Learning.

## 1. Environment Setup & Imports
*   **Libraries**:
    *   `pandas`, `numpy`: For data manipulation and numerical calculations.
    *   `requests`: To fetch real-world market data from the **Awattar API**.
    *   `matplotlib`, `seaborn`: For visualizing price trends, load profiles, and model comparisons.
    *   `sklearn` (Scikit-learn): For the Machine Learning components (Random Forest, KNN) to test price predictability.
    *   `IPython.display`: To render dynamic Markdown conclusions at the end of the run.

## 2. Data Acquisition (`fetch_awattar_data`)
This function creates the foundation of the simulation by fetching historical **Day-Ahead Market Prices** from the Awattar API (German market).
*   **Historical Range**: It is configured to fetch data going back **6 years** to ensure a robust dataset covering 2022 (crisis year) and distinct seasonal patterns.
*   **Chunking Strategy**: To avoid API timeouts or rate limits, data is fetched in **100-day chunks**.
*   **Robustness**: Includes a `retry` mechanism (up to 3 times) with a `5-second` backoff if the API returns a '429 Rate Limited' error.
*   **Preprocessing**: Converts timestamps from milliseconds to Python `datetime` objects and converts prices from `€/MWh` to `€/kWh` (dividing by 1000).

## 3. Synthetic Load Profile (`generate_synthetic_load`)
Since we don't have a real smart meter user file, we synthesize a realistic load profile for a typical household (~10 kWh/day).
*   **Logic**: Uses a weighted probability array for each hour of the day (0-23) to define usage probability.
    *   **Peaks**: Defines weighting spikes in the **Morning (06-09)** and **Evening (18-22)** matching standard residential behavior.
    *   **Night**: Low usage weights during sleep hours.
*   **Noise**: Adds Gaussian noise `np.random.normal()` to make the data look organic and less robotic.
*   **Scaling**: Normalizes the curve so the total area under the curve equals ~10 kWh per day.

## 4. Tariff Model Definitions
We define the pricing formulas for the four competing models.
*   **Parameters**:
    *   `STATIC_RATE`: **0.35 €/kWh** (Average fixed contract).
    *   `MARGIN`: **0.03 €/kWh** (Operational margin passed to vendor).
    *   `CAP_PRICE`: **0.40 €/kWh** (Protection ceiling for the Capped model).
    *   `SMOOTHING_WINDOW`: **3 Hours** (Rolling average window).

*   **Models**:
    1.  **Static**: Fixed price at 0.35 €/kWh regardless of market conditions.
    2.  **Pass-through**: `Market_Price + Margin`. Highly transparent but volatile.
    3.  **Capped**: `Min(Pass-through, Cap)`. Protects users from extreme spikes (like in 2022) but caps vendor profit during those times.
    4.  **Smoothed**: `Rolling_Avg(Market, 3h) + Margin`. Dampens sudden price spikes, providing a more stable "average" price while tracking market trends.

## 5. Simulation & Metrics
This loop calculates the financial outcome for each model.
*   **Customer Cost**: `Sum(Load_kWh * Hourly_Tariff_Price)`. (What user pays).
*   **Volatility**: `Std_Dev(Hourly_Tariff_Price)`. (How "jumpy" the price is).
*   **Vendor Profit**: Uses a **Simplified Proxy** formula as requested:
    *   `Profit = (Tariff_Price_Charged - Actual_Market_Price_Cost) * Load`
    *   This measures the spread the vendor captures on every kWh sold.

## 6. Model Selection Logic (Stability-Adjusted)
Rather than just picking the cheapest model (which might bankrupt the vendor) or the most profitable (which might be too volatile for users), we use a **Balanced Approach**.
*   **Filter**: First, discard any model that doesn't offer at least **20% savings** compared to the Static Baseline.
*   **Selection Metric**: **Stability Efficiency Score**.
    *   `Score = Vendor Profit / Volatility`
*   **Why?**: This metric favors models that deliver **healthy profits** but penalizes those with **high volatility**.
*   **Result**: This logic mathematically tends to select the **Smoothed Tariff** because it maintains the profit margin (like pass-through) but significantly reduces the volatility denominator, resulting in a higher score.

## 7. Theoretical Optimization
A "What-If" analysis that runs after the main simulation.
*   It tests margins from 0.02 to 0.15 €/kWh to see if there is a theoretical "Sweet Spot" that could generate even *more* profit while still keeping customer savings above 20%.
*   This is purely informational and doesn't override the selected Best Model.

## 8. Machine Learning Comparison
We test if the selected "Best Model" (likely the Smoothed one) is predictable. High predictability allows smart batteries to charge at optimal times.
*   **Features**:
    *   `Hour of Day`: Captures daily renewable cycles (solar dip at night).
    *   `Day of Week`: Captures lower industrial demand on weekends.
    *   `Lag_1h`: The price of the previous hour (auto-regressive).
*   **Models**:
    *   **Random Forest**: A complex ensemble method good at capturing non-linear relationships.
    *   **KNN (K-Nearest Neighbors)**: A simpler instance-based learner.
*   **Metric**: **RMSE** (Root Mean Squared Error). Lower is better. The code prints which one predicts the tariff price more accurately.

## 9. Dynamic Conclusion
Finally, a Python code cell generates a **Markdown report** on the fly.
*   It populates the text with the *actual* numbers from the run (e.g., "Customer Savings: 23.5%").
*   It explicitly states *why* the winner was chosen (e.g., "Highest Balance of Profit & Stability").
*   It summarizes the ML performance, advising which algorithm would be best for a smart home system using this tariff.
