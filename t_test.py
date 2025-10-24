import pandas as pd
from scipy.stats import pearsonr

def t_test(csv_path):
    """
    Load the accuracy scores for each dataset, and perform a Student's T-test on them
    """
    df = pd.read_csv(csv_path)
    print("Performing a T-test on the following dataset, " +
          "which compares the accuracy scores of the two models:")
    print(df)
    print("")
    r, p = pearsonr(df["acc_val"], df["acc_v2"])
    print(f"r value: {r:.10f}, p value: {p:.10f}")
    if p < 0.05:
        print("Reject the null hypothesis that these two sets of accuracy scores " +
              "are samples of the same distribution")
    else:
        print("Fail to reject the null hypothesis that these two sets of accuracy scores " +
              "are samples of the same distribution")

t_test("./results/eva02_per_class_gap.csv")
