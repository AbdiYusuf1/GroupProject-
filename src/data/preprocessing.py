'''
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
    '''
import pandas as pd                         #reads dataset to data frame
from dataclasses import dataclass           #makes a container object to return multiple outputs cleanly
from typing import List, Tuple              #type hints for readability
import os                                   #creates folders and builds paths
import json                                 #save feature to .json file
import joblib                               #saves fitted sklearn pattern

from __future__ import annotations

#sklearn imports: tools to split data and build preprocessing pipelines
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


#german data has no headings so this is to label columns
German_Column_Names: List[str] = [
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

#treat as numeric -> imputation + scaling
Numeric_Colums: List[str] = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "dependents",
]

Target_Column = "target"

@dataclass(frozen = True)                   #frozen to make immutable
class PreparedData:
    x_train_p: object                       #processed feature matrices after preprocessing
    x_test_p: object
    y_train: pd.Series                      #labels
    y_test: pd.Series
    feature_names: List[str]                #names of processed collumns. needed for reporting
    preprocessor: ColumnTransformer         #fitted sklearn transformer

def load_german_credit(path: str = "raw_data/german.data") -> pd.DataFrame:
    '''
    Load German Credit data from whitespace separated file with no headers.
    Target: 1 = Good, 2 = Bad (convert to 0/1 later)
    '''
    df = pd.read_csv(path, sep = r"\s+", header = None)     #reads the file
    if df.shape[1] != len(German_Column_Names):
        raise ValueError(
            f"Unexpected column count. Expected {len(German_Column_Names)}, got {df.shape[1]}"
        )                                   #ensures the file has the correct number of columns
    df.columns = German_Column_Names
    return df                               #outputs a dataframe with readable columns

def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    '''
    creates sklearn preprocessing pipeline

    Build an sklearn ColumnTransformer that:
        - imputes and scales numeric columns
        - imputes and one-hot encodes categorical columns
    Returns: (preprocessor, numeric_cols, categorical_cols)
    '''
    missing_numeric = [c for c in Numeric_Colums if c not in df.columns]
    if missing_numeric:
        raise ValueError(f"Numeric columns not found in dataframe: {missing_numeric}")      #checks column exists
    
    #anyhting not numeric or target is categorical
    categorical_cols = [c for c in df.columns if c not in Numeric_Colums + [Target_Column]]


    #numeric pipeline: fills missing values with the median
    #                  standardise to mean 0, std 1
    numeric_pipe = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


    #categorical pipeline: Fill missing with the most common category
    #                      One-Hot encodes (turn categories to 0/1 columns)
    categorical_pipe = Pipeline(
        steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),     #handle unknown prevents crashes if a category appears that wasn't in training
        ]
    )

    #applies the right pipeline to the correct column
    preprocessor = ColumnTransformer(
        transformers = [
            ("num", numeric_pipe, Numeric_Colums),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder = "drop",
    )

    return preprocessor, Numeric_Colums, categorical_cols
    #outputs a fitted transformer that can convert raw dataframes into matrices

def prepare_data(
        path: str = "raw_data/german.data",
        test_size: float = 0.02,
        random_state: int = 42,
        save_artifacts: bool = True,
        artifacts_dir: str = "artifacts",
) -> PreparedData:
    '''
    main function for baseline training and PyTorch work
    End to end preparation:
    - load dataset
    - convert target to 0/1. 1=bad
    - train/test split (stratisfied)
    -fir preprocessor on train; transform train/test
    -optionally save artifacts (preprocessor + feature names)
    '''
    df = load_german_credit(path)

    # convert target: 1 = good, 2 = bad -> 0 = good, 1 = bad
    df[Target_Column] = (df[Target_Column] == 2).astype(int)

    x = df.drop(columns = [Target_Column])      #drops target from features
    y = df[Target_Column]

    #split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y, #keeps the good/bad ratio similar in train and test
    )

    #fit on training set only
    preprocessor, _, _ = build_preprocessor(df)

    x_train_p = preprocessor.fit_transform(x_train)     #transform train/test into numeric matrices
    x_test_p = preprocessor.transform(x_test)

    #after one-hot encoding, there are more than 20 features
    feature_names = preprocessor.get_feature_names_out().tolist()

    if save_artifacts:
        #saves for reproducability - we can always load the same preprocessor
        os.makedirs(artifacts_dir, exist_ok=True)
        joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
        with open(os.path.join(artifacts_dir, "feature_names.json"), "w", encoding="utf-8") as f:
                  json.dump(feature_names, f, indent = 2)

    return PreparedData(                #return everything in one object
         x_train_p = x_train_p,
         x_test_p = x_test_p,
         y_train = y_train,
         y_test = y_test,
         feature_names = feature_names,
         preprocessor = preprocessor,
    )

if __name__ == "__main__":
    prepared = prepare_data(save_artifacts=True)
    print("Train shape:", getattr(prepared.X_train_p, "shape", None))
    print("Test shape: ", getattr(prepared.X_test_p, "shape", None))
    print("Train bad rate:", float(prepared.y_train.mean()))
    print("Test bad rate: ", float(prepared.y_test.mean()))
    print("Num features after encoding:", len(prepared.feature_names))