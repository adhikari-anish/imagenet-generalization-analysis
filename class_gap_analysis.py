import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("eva02_per_class_gap.csv")
df["drop_pct"] = 100 * (df["acc_val"] - df["acc_v2"])
df_sorted = df.sort_values("drop_pct", ascending=False)

print("Top 10 classes with biggest drop:")
print(df_sorted.head(10))
print("\nTop 10 classes with smallest / negative drop:")
print(df_sorted.tail(10))

# --- Bar charts ---
plt.figure(figsize=(10,5))
plt.bar(range(20), df_sorted["drop_pct"].head(20))
plt.xticks(range(20), df_sorted["class"].head(20), rotation=45, ha="right")
plt.title("Top-20 classes with largest accuracy drop (val → V2)")
plt.ylabel("Drop percentage points")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
plt.bar(range(20), df_sorted["drop_pct"].tail(20))
plt.xticks(range(20), df_sorted["class"].tail(20), rotation=45, ha="right")
plt.title("Top-20 classes least affected or improved on V2")
plt.ylabel("Drop percentage points")
plt.tight_layout()
plt.show()
