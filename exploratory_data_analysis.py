import pandas as pd

# Load your dataset
df = pd.read_csv("./caiso-electricity.csv")  # update path if needed

# Select only relevant columns
selected_columns = ["NP-15 LMP", "SP-15 LMP", "ZP-26 LMP"]

# Compute descriptive statistics
desc_stats = df[selected_columns].describe().round(2)

# Display the table
print(desc_stats)


# %% Corellations
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select the features
X = df[["NP-15 LMP", "SP-15 LMP", "ZP-26 LMP"]]

# Add constant (intercept) for statsmodels if needed
# X = sm.add_constant(X)

# Calculate VIF for each feature
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif_data["Tolerance"] = 1 / vif_data["VIF"]

print(vif_data)

# %% correlation heatmap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select only the LMP columns
region_df = df[["NP-15 LMP", "SP-15 LMP", "ZP-26 LMP"]]

# Compute correlation matrix
corr = region_df.corr()

# Plot heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of CAISO Regional Electricity Demand')
plt.tight_layout()
plt.show()
# %% time plotings
import matplotlib.pyplot as plt

# Ensure 'timestamp' is datetime type
df['Timestamp'] = pd.to_datetime(df['UTC Timestamp (Interval Ending)'])

# Set timestamp as the index
df.set_index('Timestamp', inplace=True)

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df['NP-15 LMP'], label='NP-15', alpha=0.8)
plt.plot(df['SP-15 LMP'], label='SP-15', alpha=0.8)
plt.plot(df['ZP-26 LMP'], label='ZP-26', alpha=0.8)

plt.title('CAISO Regional Electricity Demand Over Time')
plt.xlabel('Time')
plt.ylabel('Energy ($/MWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
