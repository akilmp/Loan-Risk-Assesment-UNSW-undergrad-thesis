# preprocessing.py:
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    """
    Loads a dataset from a specified CSV file.

    Parameters:
    - filepath (str): The path to the CSV file to be loaded.

    Returns:
    - DataFrame: A pandas DataFrame containing the loaded data.
    """
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Cleans the provided DataFrame by replacing specified placeholders with NaN and uses forward filling
    for handling missing values.

    Parameters:
    - df (DataFrame): The DataFrame to clean.

    Returns:
    - DataFrame: The cleaned DataFrame with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns: fill with mode or a designated category
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)  # Using mode, or could use a category like 'Unknown'

    return df

def encode_features(df):
    # Ensure all categorical features are encoded
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string in case of mixed types
    return df

def split_features_target(df, target_col_name):
    """
    Splits the DataFrame into features and target based on the target column name.

    Parameters:
    - df: DataFrame, the complete dataset including the target column.
    - target_col_name: str, the name of the column in the DataFrame that is the target for prediction.

    Returns:
    - X: DataFrame, contains all the columns except the target.
    - y: Series, contains the values of the target column.
    """
    # Features are all columns except the target
    X = df.drop(columns=[target_col_name])
    # Target is the specified column
    y = df[target_col_name]
    return X, y
