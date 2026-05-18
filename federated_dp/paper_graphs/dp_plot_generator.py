import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('dp.csv')

# 2. Define explicit colors for each noise multiplier (sigma)
color_mapping = {
    0.20: 'tab:blue',
    0.15: 'tab:green',
    0.10: 'tab:orange',
    0.05: 'tab:red'
}

# 3. Define explicit line styles for each client
style_mapping = {
    'client0': '-',   # Solid
    'client1': '--',  # Dashed
    'client2': ':',   # Dotted
}

# Get a sorted list of unique noise multipliers present in the dataset (descending order)
unique_noises = [0.20, 0.15, 0.10, 0.05]
num_subplots = len(unique_noises)

# 4. Initialize 5 stacked subplots that share the same X-axis scale
fig, axes = plt.subplots(nrows=num_subplots, ncols=1, figsize=(11, 14), sharex=True, dpi=300)

plt.rcParams.update({"font.size": 14})

# Fallback wrapper if only 1 noise multiplier exists
if num_subplots == 1:
    axes = [axes]

# 5. Populate individual subplots
for idx, noise in enumerate(unique_noises):
    ax = axes[idx]
    
    # Filter dataset for only this subplot's specific sigma level
    sub_df = df[df['noise multiplier'] == noise]
    
    # Plot each client's timeline
    for client_id, group in sub_df.groupby('client id'):
        group = group.sort_values('round')
        
        # Pull stylistic values
        line_color = color_mapping.get(round(noise, 2), 'tab:gray')
        line_style = style_mapping.get(client_id, '-')
        
        ax.plot(
            group['round'], 
            group['epsilon'], 
            color=line_color, 
            linestyle=line_style,
            alpha=0.7, 
            linewidth=2.5, 
            label=client_id  # Cleans up label so it just says client0, client1, etc.
        )
        
    # Apply configurations unique to each individual window
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='--', alpha=0.4)
    
    # Place a clean, individual client legend inside the top right corner 
    ax.legend(loc="upper right", fontsize=14, title=f"$\sigma$ = {noise:.2f}", title_fontsize=14)
    ax.set_ylabel('Privacy Budget ($\epsilon$)', fontsize=14)

# 6. Formatting final shared axis labels (applied strictly to the last graph)
axes[-1].set_xlabel('Round', fontsize=14)

# Adjust layouts to eliminate overlapping margins
plt.tight_layout()

# Save the multi-panel layout
plt.savefig('privacy_budget.png', dpi=300)