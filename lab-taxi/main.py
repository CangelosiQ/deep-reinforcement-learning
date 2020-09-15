from agent import Agent
from monitor import interact
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_rewards(l, label=""):
    plt.plot(l, label=label)
    plt.xlabel("Episodes")
    plt.ylabel("Avg. Reward")


plt.figure(figsize=(15,5))
df = pd.DataFrame()
for alpha in [0.05, 0.1, 0.15]:
    for gamma in [0.8, 0.9, 1]:
        env = gym.make('Taxi-v2')
        agent = Agent(alpha=alpha, gamma=gamma)
        avg_rewards, best_avg_reward = interact(env, agent)
        _df = pd.DataFrame({"avg_rewards":avg_rewards})
        _df["gamma"] = gamma
        _df["alpha"] = alpha
        _df["best_avg_reward"] = best_avg_reward
        df = df.append(_df, ignore_index=True)
        plot_rewards(avg_rewards, label=f"alpha={alpha}, gamma={gamma}, best={best_avg_reward}")
plt.legend()


plt.figure(figsize=(15, 5))
for alpha in np.unique(df["alpha"]):
    for gamma in np.unique(df["gamma"]):
        data = df[(df["alpha"]==alpha)&(df==gamma)]
        plt.plot(data["avg_rewards"], label=f"alpha={alpha}, gamma={gamma}, best={data['best_avg_reward'].max()}")
plt.legend()

plt.figure(figsize=(15, 5))
sns.lineplot(y=df["avg_rewards"], hue=df["alpha"], markers=df["gamma"])

plt.show()
