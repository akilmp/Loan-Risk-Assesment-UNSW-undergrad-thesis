# training.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import logging


def train_decision_tree(X_train, y_train):
    """
    Trains a decision tree classifier with pruning parameters.

    Parameters:
    - X_train: DataFrame, training feature data.
    - y_train: Series, training target data.

    Returns:
    - clf: The trained decision tree classifier.
    """
    # Define the decision tree classifier with pruning parameters
    clf = DecisionTreeClassifier(
        max_depth=5,  # Limits the depth of the tree
        min_samples_split=50,  # Minimum number of samples required to consider a split
        min_samples_leaf=25,  # Minimum samples in a leaf node
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf



def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained decision tree model using accuracy and prints the classification report.

    Parameters:
    - model: The trained decision tree classifier.
    - X_test: DataFrame, testing feature data.
    - y_test: Series, testing target data.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
