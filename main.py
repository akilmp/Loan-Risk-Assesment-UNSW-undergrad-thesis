
# Import necessary libraries and modules from other files
# These files need to be created with the appropriate functions
from preprocessing import load_data, clean_data, encode_features, split_features_target
from training import train_decision_tree, evaluate_model
from rule_extraction import extract_rules, simplify_rules
from visualization import plot_feature_importance, generate_density_plot
from sklearn.model_selection import train_test_split
from visualization import visualize_rules, visualize_rule_effect
from training import train_decision_tree, evaluate_model
import random
import numpy as np
import logging
from visualization import visualize_feature_importance, visualize_feature_hexbin_with_thresholds, visualize_feature_hexbin, visualize_rule_with_subplots, visualize_confusion_matrix, visualize_condition_impact_on_outcome, visualize_condition_probability, visualize_condition_effect
from tabulate import tabulate

def setup():
    """
    Sets up necessary configurations including random seeds and logging to ensure reproducibility
    and traceability of the run. This function configures the random seed for numpy and random libraries,
    and sets up basic logging for the application.
    """
    random.seed(42)
    np.random.seed(42)
    

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Setup completed message
    logger.info("Setup completed.")





def load_and_preprocess_data(filepath):
    """
    Load the dataset from the specified file path and preprocess it.

    Parameters:
    - filepath: str, the path to the CSV file containing the dataset.

    Returns:
    - df: DataFrame, the preprocessed dataset.
    """
    logger = logging.getLogger()
    
    # Load the data
    logger.info("Loading the data from the file.")
    df = load_data(filepath)
    
    # Clean the data
    logger.info("Cleaning the data.")
    df = clean_data(df)
    
    # Encode categorical features
    logger.info("Encoding categorical features.")
    df = encode_features(df)
    
    return df

def train_and_evaluate_model(X, y):
    """
    Trains the decision tree model and evaluates its performance.

    Parameters:
    - X: DataFrame, feature data for training and testing.
    - y: Series, target data for training and testing.

    Returns:
    - model: The trained decision tree model.
    - X_train, X_test, y_train, y_test: The split data sets for further use.
    """
    logger = logging.getLogger()
    logger.info("Splitting the data into train and test sets.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Training the decision tree model.")
    model = train_decision_tree(X_train, y_train)
    
    logger.info("Evaluating the model.")
    evaluate_model(model, X_test, y_test)

    return model, X_train, X_test, y_train, y_test


def display_rules_as_table(rules):
    """
    Displays the rules in a table format, aligning columns and headers for better readability.
    
    Parameters:
    - rules: List of simplified rule dictionaries.
    """
    table = []
    for idx, rule in enumerate(rules):
        conditions = ', '.join([f"{cond['feature']} {cond['inequality']} {cond['threshold']}" for cond in rule['conditions']])
        table.append([idx + 1, conditions, rule['outcome'], f"{rule['confidence']:.2f}", rule['support']])
    
    headers = ["Rule Number", "Conditions", "Outcome", "Confidence", "Support"]
    print(tabulate(table, headers=headers, tablefmt="pretty", numalign="center", stralign="center"))

def modify_and_test_rule(df, model, rules):
    display_rules_as_table(rules)
    rule_index = int(input("Enter the rule number to base modifications on: ")) - 1
    original_rule = rules[rule_index]

    # Display current rule details
    print("Current Rule Conditions:")
    for idx, cond in enumerate(original_rule['conditions'], start=1):
        print(f"{idx}. {cond['feature']} {cond['inequality']} {cond['threshold']}")

    # Allow user to pick a condition to modify
    condition_index = int(input("Enter the condition number to modify: ")) - 1
    condition_to_modify = original_rule['conditions'][condition_index]

    # Get new condition details from user
    new_feature = input(f"Enter new feature (current: {condition_to_modify['feature']}): ") or condition_to_modify['feature']
    new_inequality = input(f"Enter new inequality '<=' or '>' (current: {condition_to_modify['inequality']}): ") or condition_to_modify['inequality']
    new_threshold = float(input(f"Enter new threshold (current: {condition_to_modify['threshold']}): ") or condition_to_modify['threshold'])

    # Create a new rule by copying the original and modifying the selected condition
    new_rule = dict(original_rule)  # Make a shallow copy of the rule
    new_rule['conditions'] = [dict(cond) for cond in original_rule['conditions']]  # Deep copy conditions
    new_rule['conditions'][condition_index] = {'feature': new_feature, 'inequality': new_inequality, 'threshold': new_threshold}

    # Append the new, modified rule to the rules list
    rules.append(new_rule)

    # Display the updated rules table with the newly added rule
    print("Rules Table Updated with New Modified Rule:")
    display_rules_as_table(rules)

def menu(df, model, rules):
    while True:
        print("\nMenu:")
        print("1 - Display Rules as Table")
        print("2 - Graph a Rule")
        print("3 - Display Feature Importance")
        print("4 - Modify and Test a Rule")
        print("5 - Exit")
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            display_rules_as_table(rules)
        elif choice == '2':
            rule_index = int(input("Enter the rule number to graph: ")) - 1
            selected_rule = rules[rule_index]
            features_of_interest = set([cond['feature'] for cond in selected_rule['conditions']])
            for feature in features_of_interest:
                conditions = [cond for cond in selected_rule['conditions'] if cond['feature'] == feature]
                visualize_feature_hexbin_with_thresholds(df, feature, conditions=conditions)
        elif choice == '3':
            print("Feature Importance:")
            visualize_feature_importance(model, df.columns[:-1])  # assuming last column is the target
        elif choice == '4':
            modify_and_test_rule(df, model, rules)
        elif choice == '5':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please choose again.")


        
def main():
    setup()
    filename = input("Please enter the filename of the dataset (e.g., creditcard_2007_2018.csv): ")
    df = load_and_preprocess_data(filename)
    if df is None:
        print("Failed to load data. Please check the file name and try again.")
        return
    X, y = split_features_target(df, 'Outcome')
    model, X_train, X_test, y_train, y_test = train_and_evaluate_model(X, y)

    rules = extract_rules(model.tree_, X.columns)
    simplified_rules = simplify_rules(rules)

    # Starting the menu
    menu(df, model, simplified_rules)

if __name__ == "__main__":
    main()




def extract_and_visualize_rules(model, X_train, feature_names, outcome_col):
    """
    Extracts decision rules from a trained decision tree model, simplifies them,
    ranks them by predictive power, and visualizes the top rules individually.

    Parameters:
    - model: The trained decision tree model.
    - X_train: The training dataset used for the decision tree model, as a DataFrame.
    - feature_names: List of str, names of the features used to train the model.
    - outcome_col: str, the name of the outcome column in the dataset.

    """
    logger = logging.getLogger()

    # Extract rules from the model
    logger.info("Extracting rules from the decision tree.")
    rules = extract_rules(model.tree_, feature_names)

    # Simplify the rules and select the top ones for visualization
    logger.info("Simplifying and selecting the top rules based on predictive power.")
    simplified_rules = simplify_rules(rules)
    
    # We will visualize only the top N rules for brevity and clarity
    top_rules = simplified_rules[:5]  # Change the number as needed

    # Visualize each of the top rules
    for i, rule in enumerate(top_rules, start=1):
        logger.info(f"Visualizing Rule {i}: {rule['rule']}")
        visualize_rule_effect(X_train, rule, outcome_col)


def generate_plots(model, df, feature_names):
    """
    Generates and displays various plots to help understand the model and data.

    Parameters:
    - model: The trained decision tree model.
    - df: DataFrame, the entire preprocessed dataset used for model training and evaluation.
    - feature_names: List of str, the names of the features used in the model, excluding the target.

    Returns:
    - None
    """
    logger = logging.getLogger()

    # Plot feature importance
    logger.info("Generating feature importance plot.")
    plot_feature_importance(model, feature_names)

    # Generate density plot
    logger.info("Generating density plot for interest rate and loan amount.")
    # Assuming 'int_rate_n' and 'loan_amnt_n' are columns in your dataframe and 'Outcome' is the target
    generate_density_plot(df, 'int_rate_n', 'loan_amnt_n', 'Outcome')