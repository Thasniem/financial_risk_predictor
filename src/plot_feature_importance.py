import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("outputs/plots/feature_importance.csv")
df = df.sort_values(by="Importance", ascending=False).head(20)

# Plot setup
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")
bars = plt.barh(df["Feature"], df["Importance"], color="#69b3a2", edgecolor='black')

# Labels and titles
plt.xlabel("Importance", fontsize=12)
plt.title("Top 20 Feature Importances", fontsize=16, fontweight='bold')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.gca().invert_yaxis()

# Add value labels *inside* or close to bars
for bar in bars:
    width = bar.get_width()
    plt.text(width - 0.005, bar.get_y() + bar.get_height()/2,
             f"{width:.3f}", ha='right', va='center', fontsize=9, color='white', fontweight='bold')

# Layout and save
plt.tight_layout()
plt.savefig("outputs/plots/feature_importance_cleaned.png", dpi=300)
plt.show()
