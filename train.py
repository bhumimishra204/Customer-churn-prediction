
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def main():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Download the Telco Customer Churn CSV and place it in the data folder.")
        return

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nColumn information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nChurn distribution:")
    print(df["Churn"].value_counts())


if __name__ == "__main__":
    main()
