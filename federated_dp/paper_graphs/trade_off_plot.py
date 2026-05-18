import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
# Ensure '2clip.csv' contains the header row: 
# Noise Multiplier,Macro F1,PR-AUC,Max Client Privacy Budget
df = pd.read_csv('2clip.csv')

# Sort by Noise Multiplier to ensure lines connect correctly
df = df.sort_values(by='Noise Multiplier')

# 2. Initialize the plot
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
plt.rcParams.update({"font.size": 14})

# 3. Plot Macro F1 and PR-AUC on the left Y-axis
line1, = ax1.plot(df['Noise Multiplier'], df['Macro F1'], color='tab:blue', marker='o', label='Macro F1')
line2, = ax1.plot(df['Noise Multiplier'], df['PR-AUC'], color='tab:cyan', marker='s', label='PR-AUC')

ax1.set_xlabel('Noise Multiplier ($\sigma$)', fontsize=14)
ax1.set_ylabel('Performance Metrics', fontsize=14)
ax1.tick_params(axis='both', labelsize=14)
ax1.tick_params(axis='y')
ax1.grid(True, linestyle='--', alpha=0.6)

# 4. Create a second Y-axis sharing the same X-axis for Privacy Budget
ax2 = ax1.twinx()

# Using a log scale since max budget ranges from ~311 to ~685,431,178
line3, = ax2.plot(df['Noise Multiplier'], df['Max Client Privacy Budget'], color='tab:red', marker='^', linestyle='--', label='Max $\epsilon$')

# Changed color and labelcolor to 'black' here:
ax2.set_ylabel('Max Client Privacy Budget ($\epsilon$)', color='black', fontsize=14)
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_yscale('log')  # Log scale handles the huge exponential jump in epsilon safely

# 5. Handle legends from both axes
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper right')

# 6. Title and layout adjustments
fig.tight_layout()

# Save and show the plot
plt.savefig('privacy_utility_tradeoff_2c.png', dpi=300)