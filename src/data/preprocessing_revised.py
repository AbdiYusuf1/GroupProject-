from __future__ import annotations

from dataclasses import dataclass                   #makes a container object to return multiple outputs cleanly
from pathlib import Path
from typing import Any, List, Tuple                 #type hints for readability
import os                                           #creates folders and builds paths
import json                                         #save feature to .json file
import joblib                                       #saves fitted sklearn pattern
import pandas as pd                                 #reads dataset to data frame

#sklearn imports: tools to split data and build preprocessing pipelines
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#german data has no headings so this is to label columns
GERMAN_COLUMN_NAMES: List[str] = [
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
    "target",
]

#treat as numeric -> imputation + scaling
NUMERIC_COLUMNS: List[str] = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age",
    "existing_credits",
    "dependents",
]

TARGET_COLUMN = "target"


@dataclass(frozen=True)                             #frozen to make immutable
class PreparedData:
    x_train_p: Any                                  #processed feature matrices
    x_val_p: Any
    x_test_p: Any
    y_train: pd.Series                              #labels
    y_val: pd.Series
    y_test: pd.Series
    feature_names: List[str]                        #names of processed collumns. needed for reporting
    preprocessor: ColumnTransformer                 #fitted sklearn transformer

    #Old code may use upper or lower case
    @property
    def X_train_p(self) -> Any:
        return self.x_train_p

    @property
    def X_val_p(self) -> Any:
        return self.x_val_p

    @property
    def X_test_p(self) -> Any:
        return self.x_test_p


def _default_data_path() -> Path:
    """
    Try a few sensible locations for german.data.
    """
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] / "raw_data" / "german.data",  # project_root/raw_data/german.data
        here.parents[1] / "raw_data" / "german.data",  # src/raw_data/german.data
        Path.cwd() / "raw_data" / "german.data",       # current working dir/raw_data/german.data
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_german_credit(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """
    Load the German Credit dataset from a whitespace-separated file with no headers.
    Original target values are 1 = good, 2 = bad.
    """
    resolved_path = Path(path) if path is not None else _default_data_path()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Could not find german.data. Expected it at "
            f"'{resolved_path}'. Place the file in raw_data/german.data or pass a path explicitly."
        )

    df = pd.read_csv(resolved_path, sep=r"\s+", header=None)
    if df.shape[1] != len(GERMAN_COLUMN_NAMES):
        raise ValueError(
            f"Unexpected column count. Expected {len(GERMAN_COLUMN_NAMES)}, got {df.shape[1]}"
        )

    df.columns = GERMAN_COLUMN_NAMES
    return df


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a preprocessing pipeline:
      - numeric columns: median imputation + standardisation
      - categorical columns: mode imputation + one-hot encoding
    """
    missing_numeric = [c for c in NUMERIC_COLUMNS if c not in df.columns]
    if missing_numeric:
        raise ValueError(f"Numeric columns not found in dataframe: {missing_numeric}")

    categorical_columns = [
        c for c in df.columns if c not in NUMERIC_COLUMNS + [TARGET_COLUMN]
    ]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLUMNS),
            ("cat", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
    )

    return preprocessor, NUMERIC_COLUMNS, categorical_columns


def prepare_data(
    path: str | os.PathLike[str] | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    save_artifacts: bool = False,
    artifacts_dir: str | os.PathLike[str] = "artifacts",
) -> PreparedData:
    """
    End-to-end data preparation:
      - load dataset
      - convert target to 0/1 (0 = good, 1 = bad)
      - split into train/validation/test with stratification
      - fit preprocessor on training data only
      - transform train/validation/test
      - optionally save preprocessing artefacts
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    df = load_german_credit(path)
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == 2).astype(int)

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # val_size is specified as a fraction of the full dataset, so convert it to
    # a fraction of the remaining train_full split.
    relative_val_size = val_size / (1.0 - test_size)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_full,
    )

    preprocessor, _, _ = build_preprocessor(x_train.assign(**{TARGET_COLUMN: y_train}))

    x_train_p = preprocessor.fit_transform(x_train)
    x_val_p = preprocessor.transform(x_val)
    x_test_p = preprocessor.transform(x_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    if save_artifacts:
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, artifacts_path / "preprocessor.joblib")
        with open(artifacts_path / "feature_names.json", "w", encoding="utf-8") as file:
            json.dump(feature_names, file, indent=2)

    return PreparedData(
        x_train_p=x_train_p,
        x_val_p=x_val_p,
        x_test_p=x_test_p,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        preprocessor=preprocessor,
    )


if __name__ == "__main__":
    prepared = prepare_data(save_artifacts=True)
    print("Train shape:", getattr(prepared.x_train_p, "shape", None))
    print("Validation shape:", getattr(prepared.x_val_p, "shape", None))
    print("Test shape:", getattr(prepared.x_test_p, "shape", None))
    print("Train bad rate:", float(prepared.y_train.mean()))
    print("Validation bad rate:", float(prepared.y_val.mean()))
    print("Test bad rate:", float(prepared.y_test.mean()))
    print("Number of features after encoding:", len(prepared.feature_names))
