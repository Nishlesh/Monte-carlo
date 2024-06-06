Let's process the newly provided CSV data and spot price (22,791.05). The time to expiration is \( \frac{21}{365} \) years.

### Updated Steps:

1. **Load the CSV file**.
2. **Select strikes**: Choose 3 strikes from 10 ITM to 10 OTM around the spot price.
3. **Generate all combinations**: Create combinations of buying or selling calls or puts for the selected strikes.
4. **Calculate metrics**: Compute breakeven points, probability of profit, max profit, and max profit/max loss ratio for each combination using the Black-Scholes-Merton model.
5. **Construct the decision matrix**: Populate it with calculated values.
6. **Apply TOPSIS method**: Rank these combinations using the provided weights.

### Implementing the Steps:

#### Load the Data:
```python
import pandas as pd

# Load the CSV file
file_path = '/mnt/data/option-chain-ED-NIFTY-27-Jun-2024.xlsx'
data = pd.read_excel(file_path)

# Display the data
data.head()
```

#### Selecting Strikes:
We'll choose 3 strikes each from 10 ITM to 10 OTM around the ATM strike. The ATM strike is the closest strike price to the spot price (22,791.05).

```python
# Spot price
spot_price = 22791.05

# Filter out the relevant data
calls = data[['STRIKE', 'CALLS_LTP', 'CALLS_IV']].dropna()
puts = data[['STRIKE', 'PUTS_LTP', 'PUTS_IV']].dropna()

# Find ATM strike
atm_strike = min(data['STRIKE'], key=lambda x: abs(x - spot_price))

# Select 10 ITM and 10 OTM strikes
itm_strikes = sorted([strike for strike in data['STRIKE'] if strike < atm_strike])[-10:]
otm_strikes = sorted([strike for strike in data['STRIKE'] if strike > atm_strike])[:10]
selected_strikes = itm_strikes + [atm_strike] + otm_strikes

selected_strikes
```

#### Generate All Combinations:
```python
from itertools import combinations, product

# Generate all possible combinations of buying/selling calls or puts for each selected strike
option_actions = list(product(['buy', 'sell'], repeat=3))

combinations_list = []

# Generate all combinations
for strikes_comb in combinations(selected_strikes, 3):
    for actions in option_actions:
        for option_type in ['call', 'put']:
            combination = []
            for i in range(3):
                combination.append((option_type, actions[i], strikes_comb[i]))
            combinations_list.append(combination)

# Display the number of generated combinations
len(combinations_list)
```

#### Calculate Metrics for Each Combination:
```python
from scipy.stats import norm
import numpy as np

# Black-Scholes-Merton (BSM) model
def bsm_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Placeholder values for interest rate
r = 0.01  # Risk-free interest rate (e.g., 1%)
T = 21 / 365  # Time to expiration in years

results = []

for combination in combinations_list:
    metrics = {}
    total_premium = 0
    total_max_profit = 0
    total_max_loss = 0
    total_probability_of_profit = 0

    for option in combination:
        option_type, action, strike = option
        row = data[data['STRIKE'] == strike]
        if option_type == 'call':
            IV = row['CALLS_IV'].values[0]
            LTP = row['CALLS_LTP'].values[0]
        else:
            IV = row['PUTS_IV'].values[0]
            LTP = row['PUTS_LTP'].values[0]

        S = spot_price  # Spot price
        sigma = IV / 100

        # Calculate BSM price
        option_price = bsm_price(S, strike, T, r, sigma, option_type)

        # Calculate max profit and max loss
        if action == 'buy':
            max_profit = option_price - LTP
            max_loss = LTP
        else:
            max_profit = LTP
            max_loss = option_price - LTP

        # Calculate probability of profit (simplified)
        d1 = (np.log(S / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        probability_of_profit = norm.cdf(d1) if option_type == 'call' else 1 - norm.cdf(d1)

        total_premium += option_price
        total_max_profit += max_profit
        total_max_loss += max_loss
        total_probability_of_profit += probability_of_profit

    metrics['total_premium'] = total_premium
    metrics['total_max_profit'] = total_max_profit
    metrics['total_max_loss'] = total_max_loss
    metrics['total_probability_of_profit'] = total_probability_of_profit / 3  # Average probability
    results.append(metrics)

# Display results
results[:5]
```

#### Construct Decision Matrix:
```python
# Create decision matrix
decision_matrix = np.array([
    [result['total_probability_of_profit'], result['total_max_profit'], result['total_max_profit'] / result['total_max_loss']] for result in results
])

# Display the decision matrix
decision_matrix[:5]
```

#### Apply TOPSIS Method:
```python
from sklearn.preprocessing import MinMaxScaler

# Placeholder weights (to be provided by the user)
weights = np.array([0.5, 0.3, 0.2])  # Replace with actual weights

# Normalize the decision matrix
scaler = MinMaxScaler()
normalized_matrix = scaler.fit_transform(decision_matrix)

# Calculate TOPSIS score
def topsis(matrix, weights):
    weighted_matrix = matrix * weights
    ideal_solution = np.max(weighted_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_matrix, axis=0)
    distance_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    distance_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))
    topsis_score = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    return topsis_score

# Calculate TOPSIS scores
topsis_scores = topsis(normalized_matrix, weights)

# Find the best strategy
best_index = np.argmax(topsis_scores)
best_strategy = combinations_list[best_index]

print("Best strategy:", best_strategy)
print("TOPSIS score:", topsis_scores[best_index])
```

### Full Code Implementation

Here’s the complete code:
```python
import pandas as pd
from itertools import combinations, product
from scipy.stats import norm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the Excel file
file_path = '/mnt/data/option-chain-ED-NIFTY-27-Jun-2024.xlsx'
data = pd.read_excel(file_path)

# Spot price
spot_price = 22791.05

# Filter out the relevant data
calls = data[['STRIKE', 'CALLS_LTP', 'CALLS_IV']].dropna()
puts = data[['STRIKE', 'PUTS_LTP', 'PUTS_IV']].dropna()

# Find ATM strike
atm_strike = min(data['STRIKE'], key=lambda x: abs(x - spot_price))

# Select 10 ITM and 10 OTM strikes
itm_strikes = sorted([strike for strike in data['STRIKE'] if strike < atm_strike])[-10:]
otm_strikes = sorted([strike for strike in data['STRIKE'] if strike > atm_strike])[:10]
selected_strikes = itm_strikes + [atm_strike] + otm_strikes

# Generate all possible combinations of buying/selling calls or puts for each selected strike
option_actions = list(product(['buy', 'sell'], repeat=3))
combinations_list = []

# Generate all combinations
for strikes_comb in combinations(selected_strikes, 3):
    for actions in option_actions:
        for option_type in ['call', 'put']:
            combination = []
            for i in range(3):
                combination.append((option_type, actions[i], strikes_comb[i]))
            combinations_list.append(combination)

# Black-Scholes-Merton (BSM) model
def bsm_price(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
   

 else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Placeholder values for interest rate
r = 0.01  # Risk-free interest rate (e.g., 1%)
T = 21 / 365  # Time to expiration in years

results = []

for combination in combinations_list:
    metrics = {}
    total_premium = 0
    total_max_profit = 0
    total_max_loss = 0
    total_probability_of_profit = 0

    for option in combination:
        option_type, action, strike = option
        row = data[data['STRIKE'] == strike]
        if option_type == 'call':
            IV = row['CALLS_IV'].values[0]
            LTP = row['CALLS_LTP'].values[0]
        else:
            IV = row['PUTS_IV'].values[0]
            LTP = row['PUTS_LTP'].values[0]

        S = spot_price  # Spot price
        sigma = IV / 100

        # Calculate BSM price
        option_price = bsm_price(S, strike, T, r, sigma, option_type)

        # Calculate max profit and max loss
        if action == 'buy':
            max_profit = option_price - LTP
            max_loss = LTP
        else:
            max_profit = LTP
            max_loss = option_price - LTP

        # Calculate probability of profit (simplified)
        d1 = (np.log(S / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        probability_of_profit = norm.cdf(d1) if option_type == 'call' else 1 - norm.cdf(d1)

        total_premium += option_price
        total_max_profit += max_profit
        total_max_loss += max_loss
        total_probability_of_profit += probability_of_profit

    metrics['total_premium'] = total_premium
    metrics['total_max_profit'] = total_max_profit
    metrics['total_max_loss'] = total_max_loss
    metrics['total_probability_of_profit'] = total_probability_of_profit / 3  # Average probability
    results.append(metrics)

# Create decision matrix
decision_matrix = np.array([
    [result['total_probability_of_profit'], result['total_max_profit'], result['total_max_profit'] / result['total_max_loss']] for result in results
])

# Placeholder weights (to be provided by the user)
weights = np.array([0.5, 0.3, 0.2])  # Replace with actual weights

# Normalize the decision matrix
scaler = MinMaxScaler()
normalized_matrix = scaler.fit_transform(decision_matrix)

# Calculate TOPSIS score
def topsis(matrix, weights):
    weighted_matrix = matrix * weights
    ideal_solution = np.max(weighted_matrix, axis=0)
    negative_ideal_solution = np.min(weighted_matrix, axis=0)
    distance_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal_solution) ** 2, axis=1))
    distance_to_negative_ideal = np.sqrt(np.sum((weighted_matrix - negative_ideal_solution) ** 2, axis=1))
    topsis_score = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    return topsis_score

# Calculate TOPSIS scores
topsis_scores = topsis(normalized_matrix, weights)

# Find the best strategy
best_index = np.argmax(topsis_scores)
best_strategy = combinations_list[best_index]

print("Best strategy:", best_strategy)
print("TOPSIS score:", topsis_scores[best_index])
```

This code will calculate the breakeven points, probability of profit, maximum profit, and the max profit/max loss ratio for each option combination. It will then use the TOPSIS method to determine the best strategy based on these metrics.
