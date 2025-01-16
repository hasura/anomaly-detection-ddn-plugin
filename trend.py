import numpy as np
import matplotlib.pyplot as plt

# Given data points
weeks = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
growth_rates = np.array([0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.46, 0.8])

# Fit a linear trend line
coefficients = np.polyfit(weeks, growth_rates, 1)
trend_line = np.poly1d(coefficients)

# Extrapolate for the next 52 weeks
future_weeks = np.arange(1, 53)
predicted_growth_rates = trend_line(future_weeks)

# Calculate the number of followers
initial_followers = 1141
followers = [initial_followers]
for rate in predicted_growth_rates:
    new_followers = followers[-1] * (1 + rate / 100)
    followers.append(new_followers)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Weeks')
ax1.set_ylabel('7-Day Growth Rate (%)', color=color)
ax1.plot(future_weeks, predicted_growth_rates, '-', label='Predicted Trend Line', color=color)
ax1.scatter(weeks, growth_rates, color=color, label='Original Data Points')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Number of Followers', color=color)
ax2.plot(future_weeks, followers[1:], '-', label='Number of Followers', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Predicted 7-Day Growth Rate Trend Line and Number of Followers')
plt.grid(True)
plt.legend()
plt.show()
