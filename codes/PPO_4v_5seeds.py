import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

# Read the data from the file
df_ppo_0 = pd.read_csv("PPO4.0.csv")
df_ppo_1 = pd.read_csv("PPO4.1.csv")
df_ppo_2 = pd.read_csv("PPO4.2.csv")
df_ppo_3 = pd.read_csv("PPO4.3.csv")

# Extract columns containing reward values for each seed
ppo_reward_columns_0 = ['PPO4.0_0', 'PPO4.0_1','PPO4.0_10', 'PPO4.0_50','PPO4.0_100']
ppo_reward_columns_1 = ['PPO4.1_0', 'PPO4.1_1','PPO4.1_10', 'PPO4.1_50','PPO4.1_100']
ppo_reward_columns_2 = ['PPO4.2_0', 'PPO4.2_1','PPO4.2_10', 'PPO4.2_50','PPO4.2_100']
ppo_reward_columns_3 = ['PPO4.3_0', 'PPO4.3_1','PPO4.3_10', 'PPO4.3_50','PPO4.3_100']

# Compute the mean reward across seeds for each step
ppo_std_0 = df_ppo_0[ppo_reward_columns_0].std(axis=1)
ppo_avg_0 = df_ppo_0[ppo_reward_columns_0].mean(axis=1)

ppo_std_1 = df_ppo_1[ppo_reward_columns_1].std(axis=1)
ppo_avg_1 = df_ppo_1[ppo_reward_columns_1].mean(axis=1)

ppo_std_2 = df_ppo_2[ppo_reward_columns_2].std(axis=1)
ppo_avg_2 = df_ppo_2[ppo_reward_columns_2].mean(axis=1)

ppo_std_3 = df_ppo_3[ppo_reward_columns_3].std(axis=1)
ppo_avg_3 = df_ppo_3[ppo_reward_columns_3].mean(axis=1)

# Assuming you have data for PPO and SAC
ppo_steps = df_ppo_0['Step']

ppo_rewards_0 = ppo_avg_0
ppo_rewards_1 = ppo_avg_1
ppo_rewards_2 = ppo_avg_2
ppo_rewards_3 = ppo_avg_3

# Create a new figure
plt.figure()

# # Plot PPO data
# plt.plot(ppo_steps, ppo_rewards_0, color='blue', label='V0')
# plt.fill_between(ppo_steps, ppo_rewards_0 - ppo_std_0, ppo_rewards_0 + ppo_std_0, color='blue', alpha=0.1)

colors = ['blue', 'red', 'green', 'purple', 'orange']
labels = ['V0', 'V1', 'V2', 'V3', 'V4']

for i in range(4):
    plt.plot(ppo_steps, eval(f'ppo_rewards_{i}'), color=colors[i], label=labels[i])
    plt.fill_between(ppo_steps, eval(f'ppo_rewards_{i}') - eval(f'ppo_std_{i}'), eval(f'ppo_rewards_{i}') + eval(f'ppo_std_{i}'), color=colors[i], alpha=0.1)

plot_type='PPO 4 versions'
result=' Mean Reward'  # Mean Reward   Max Reward   Mean Success Rate

# Add labels and legend
plt.title(plot_type+result)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend(loc='lower right')

# save as PDF （dpi=600）
plt.savefig(plot_type+'.png', format='png', dpi=600)

# Show the plot
plt.show()

