Let's proceed by calculating the values for the decision matrix based on the options data you provided. Here are the steps we'll follow:

1. **Select the strikes**: Choose 3 strikes from the provided options data, 10 ITM to 10 OTM.
2. **Generate combinations**: For each of the chosen strikes, generate all possible combinations of buying and selling calls or puts.
3. **Calculate breakeven points, probability of profit, max profit/max loss, and max profit for each combination**.
4. **Construct the decision matrix**: Populate the matrix with calculated values.
5. **Use the TOPSIS method** to rank these combinations.

### Step 1: Load the data and choose the strikes
I'll load the data from the CSV file and then choose three strikes from the provided options data.

### Step 2: Generate combinations
For each chosen strike, generate all possible combinations of buying and selling calls or puts. Given the large number of possible combinations, we will focus on creating a subset to demonstrate the approach.

### Step 3: Calculate metrics
Using the Black-Scholes-Merton (BSM) model and the lognormal distribution, we will calculate the necessary metrics for each combination.

### Step 4: Construct the decision matrix
Populate the decision matrix with calculated values.

### Step 5: Apply the TOPSIS method
Using the decision matrix and user-provided weights, we will apply the TOPSIS method to rank the combinations.

Let's start by loading the data from the CSV file and selecting three strikes.

#### Loading the data:
```python
import pandas as pd

# Load the CSV file
data = pd.read_csv("/mnt/data/image.png")

# Display the first few rows of the dataframe
data.head()
```

### Choosing Strikes and Generating Combinations:
To demonstrate the approach, we will manually choose three strikes: one ITM, one ATM, and one OTM. Then, we'll generate a few combinations.

#### Define the Strikes and Generate Combinations:
```python
from itertools import combinations, product

# Choose three strikes (for demonstration purposes, we will pick manually)
chosen_strikes = [20050, 21000, 21500]

# Generate all possible combinations of buying/selling calls or puts for each chosen strike
# Each strike has two options: call and put
# Each option can be bought or sold
# Total combinations = 2 (call/put) * 2 (buy/sell) ^ 3 (number of strikes)

strike_combinations = list(combinations(chosen_strikes, 3))
option_actions = list(product(['buy', 'sell'], repeat=3))

combinations_list = []

# Generate all combinations
for strikes in strike_combinations:
    for actions in option_actions:
        for option_type in ['call', 'put']:
            combination = []
            for i in range(3):
                combination.append((option_type, actions[i], strikes[i]))
            combinations_list.append(combination)

# Display a sample of generated combinations
combinations_list[:5]
```

### Calculating Metrics for Each Combination:
We need to calculate breakeven points, probability of profit, max profit/max loss, and max profit for each combination. For this, we'll use the BSM model and lognormal distribution.

#### Calculate Metrics Using BSM Model:
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

# Calculate metrics for each combination
results = []

# Placeholder values for interest rate and time to maturity
r = 0.01  # Risk-free interest rate (e.g., 1%)
T = 30 / 365  # Time to maturity (30 days)

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
            IV = row['IV'].values[0]
            LTP = row['LTP'].values[0]
        else:
            IV = row['IV.1'].values[0]
            LTP = row['LTP.1'].values[0]

        S = row['Spot_Price'].values[0]  # Spot price (needs to be provided)
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

### Constructing the Decision Matrix:
Now we populate the decision matrix with calculated values.

#### Construct Decision Matrix:
```python
decision_matrix = np.array([
    [result['total_probability_of_profit'], result['total_max_profit'], result['total_max_profit'] / result['total_max_loss']] for result in results
])

# Display the decision matrix
print("Decision Matrix:\n", decision_matrix)
```

### Applying TOPSIS Method:
Finally, apply the TOPSIS method to rank these combinations.

#### Applying TOPSIS:
```python
from sklearn.preprocessing import MinMaxScaler

# Placeholder weights (to be provided by the user)
weights = np.array([0.5, 0.3, 0.2])

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

This code outlines the complete process from loading the data to identifying the best options strategy using the TOPSIS method. You can execute this code within your environment to get the best strategy based on the provided data.
