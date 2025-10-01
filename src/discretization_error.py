from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

n_samples = 10_000
x = np.linspace(0, 1, n_samples)
y = x * (1 - x)

plt.plot(x, y, label='y = p(1-p)', color='blue')

for i in np.arange(0.05, 1, 0.1):
    # line in x=i from y=0 to y=i(1-i)
    plt.plot([i, i], [0, i * (1 - i)], color='black', linestyle='--')

# X mark the point (0.5, 0.25)
plt.plot(0.5, 0.25, marker='x', markersize=10,
         color='green', label='Optimal price (y=0.25)')

# X mark the point (0.45, 0.45*(1-0.45))
plt.plot(0.45, 0.45 * (1 - 0.45), marker='x', markersize=10,
         color='red', label=f'Discrete price (y={0.45 * (1 - 0.45):.3f})')

plt.title('Optimal price vs Discrete optimal price')

plt.xlabel('Price')
plt.ylabel('Revenue')

plt.grid()

plt.legend()
results_dir = Path("../results/others")
results_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(results_dir / 'optimal_price_uniform.png')
plt.show()
