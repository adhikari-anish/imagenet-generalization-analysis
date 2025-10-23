import pandas as pd
import matplotlib.pyplot as plt

# Load feature drift data
df = pd.read_csv("eva02_feature_shift_val_vs_v2.csv")

# Sort by standardized shift (how many stds apart the distributions are)
df_sorted = df.sort_values("std_shift", ascending=False)

# Inspect top and bottom
print("Top 10 features with largest standardized shift:")
print(df_sorted[["feat", "std_shift"]].head(10))
# print(df_sorted.head(10))

print("\n10 most stable features:")
print(df_sorted[["feat", "std_shift"]].tail(10))
# print(df_sorted.tail(10))

# Plot
plt.figure(figsize=(10,5))
plt.bar(range(20), df_sorted["std_shift"].head(20))
plt.xticks(range(20), df_sorted["feat"].head(20), rotation=45, ha="right")
plt.ylabel("Standardized shift (σ units)")
plt.title("Top-20 Deep Features with Largest Domain Shift (val → V2)")
plt.tight_layout()
plt.show()
