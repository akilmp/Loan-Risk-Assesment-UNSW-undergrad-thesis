from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import logging
import joblib

def save_model(model, filename):
    """
    Saves the trained model to disk.

    Parameters:
    - model: the trained model object to be saved.
    - filename: str, path where to save the model.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Loads a trained model from disk.

    Parameters:
    - filename: str, path where the model is saved.

    Returns:
    - model: the loaded model object.
    """
    return joblib.load(filename)

def inspect_data(df, n=5):
    """
    Prints basic information and the first n rows of the DataFrame to inspect its structure and content.

    Parameters:
    - df: DataFrame, the data to inspect.
    - n: int, number of rows to display from the DataFrame.
    """
    print(df.info())
    print(df.head(n))

def setup_logging(log_file='project.log', level=logging.INFO):
    """
    Sets up the logging configuration.

    Parameters:
    - log_file: str, path to the log file.
    - level: logging level.
    """
    logging.basicConfig(filename=log_file, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=level)

def calculate_auc(y_true, y_pred):
    """
    Calculates the Area Under the ROC Curve (AUC) from prediction scores.

    Parameters:
    - y_true: array-like of true binary labels.
    - y_pred: array-like of target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.

    Returns:
    - auc_score: float, AUC score.
    """
    auc_score = roc_auc_score(y_true, y_pred)
    return auc_score

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets.

    Parameters:
    - X: DataFrame or array-like, feature dataset.
    - y: Series or array-like, target dataset.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, is the seed used by the random number generator.

    Returns:
    - X_train, X_test, y_train, y_test: split data sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
