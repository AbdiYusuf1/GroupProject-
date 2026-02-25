import pandas as pd

def load_german_credit(path = "raw_data/german.data"):
    #german dataset is whitespaced separated, not comma serparated
    df = pd.read_csv(path, sep = r"\s+", header = None)

    #Last column is target
    column_names = [
        "checking_account_status",
        "duration_months",
        "credit_history",
        "purpose",
        "credit_amount",
        "savings_account",
        "employment_since",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "residence_since",
        "property",
        "age",
        "other_installment_plans",
        "housing",
        "existing_credits",
        "job",
        "dependents",
        "telephone",
        "foreign_worker",
        "target"
    ]

    df.columns = column_names

    return df

if __name__ == "__main__":
    df = load_german_credit()
    print("Shape:", df.shape)
    print(df.head())
    print(df["target"].value_counts())