# generate_plot.py
import os
import matplotlib.pyplot as plt

# Make an output folder
os.makedirs("visuals", exist_ok=True)

# Example data
days = list(range(1, 11))
values = [3, 4, 6, 5, 7, 8, 8, 9, 11, 10]

# Create a simple line chart
plt.figure(figsize=(6, 4))
plt.plot(days, values, marker="o")
plt.title("Example Trend")
plt.xlabel("Day")
plt.ylabel("Value")
plt.tight_layout()

# Save as PNG (and SVG for crispness if you want)
plt.savefig(os.path.join("visuals", "plot.png"), dpi=150)
plt.savefig(os.path.join("visuals", "plot.svg"))
plt.close()
