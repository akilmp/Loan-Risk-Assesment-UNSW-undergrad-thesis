import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.tight_layout()
    plt.show()

def visualize_rule_effect(df, rule, outcome_col):
    rule_df = df.copy()
    for condition in rule['conditions']:
        feature, inequality, threshold = condition['feature'], condition['inequality'], condition['threshold']
        if inequality == "<=":
            rule_df = rule_df[rule_df[feature] <= threshold]
        else:
            rule_df = rule_df[rule_df[feature] > threshold]

    plt.figure(figsize=(12, 8))
    sns.countplot(x=outcome_col, data=rule_df, palette="viridis")
    plt.title(f"Distribution of '{outcome_col}' for data points satisfying the rule")
    plt.xlabel(outcome_col)
    plt.ylabel('Count')
    plt.show()

def generate_density_plot(df, x_feature, y_feature, hue):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=x_feature, y=y_feature, hue=hue, fill=True, palette="viridis")
    plt.title('Density Plot')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.show()

def visualize_rules(df, rules):
    for rule in rules:
        plt.figure(figsize=(12, 8))
        filtered_df = df.copy()
        
        for condition in rule['conditions']:
            if condition['inequality'] == '<=':
                filtered_df = filtered_df[filtered_df[condition['feature']] <= condition['threshold']]
            else:
                filtered_df = filtered_df[filtered_df[condition['feature']] > condition['threshold']]

        sns.kdeplot(data=df, x=condition['feature'], label='All Data', fill=True, color="blue")
        sns.kdeplot(data=filtered_df, x=condition['feature'], label='Filtered by Rule', fill=True, color="orange")
        
        plt.title(f"Rule Impact: Outcome = {rule['outcome']} with Confidence = {rule['confidence']:.2f} and Support = {rule['support']}")
        plt.legend()
        plt.xlabel(condition['feature'])
        plt.ylabel('Density')
        plt.show()

def visualize_rule_with_subplots(df, rule, outcome_col):
    num_conditions = len(rule['conditions'])
    fig, axs = plt.subplots(num_conditions, 1, figsize=(10, 5 * num_conditions), constrained_layout=True)

    if num_conditions == 1:
        axs = [axs]

    for idx, condition in enumerate(rule['conditions']):
        if condition['inequality'] == '<=':
            condition_df = df[df[condition['feature']] <= condition['threshold']]
        else:
            condition_df = df[df[condition['feature']] > condition['threshold']]

        sns.histplot(data=df, x=outcome_col, ax=axs[idx], kde=True, stat="density", label='All Data', color="blue", alpha=0.5)
        sns.histplot(data=condition_df, x=outcome_col, ax=axs[idx], kde=True, stat="density", label='Filtered by Condition', color="orange", alpha=0.5)
        axs[idx].set_title(f"Condition {idx + 1}: {condition['feature']} {condition['inequality']} {condition['threshold']}")
        axs[idx].legend()

    plt.show()

def visualize_confusion_matrix(df, rule, outcome_col):
    filtered_df = df.copy()
    for condition in rule['conditions']:
        if condition['inequality'] == '<=':
            filtered_df = filtered_df[filtered_df[condition['feature']] <= condition['threshold']]
        else:
            filtered_df = filtered_df[filtered_df[condition['feature']] > condition['threshold']]
    
    predicted_outcome = rule['outcome']
    actual_outcomes = filtered_df[outcome_col]
    predicted_outcomes = np.full_like(actual_outcomes, fill_value=predicted_outcome)

    cm = confusion_matrix(actual_outcomes, predicted_outcomes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(df[outcome_col]))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Rule')
    plt.show()

def visualize_condition_impact_on_outcome(df, rule, outcome_col):
    for condition in rule['conditions']:
        feature, threshold, inequality = condition['feature'], condition['threshold'], condition['inequality']
        condition_met = df[feature] > threshold if inequality == '>' else df[feature] <= threshold
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=df[outcome_col], y=df[feature], hue=condition_met, split=True, inner='quartile', palette="viridis")
        plt.title(f"Impact of {feature} {inequality} {threshold} on {outcome_col}")
        plt.xlabel('Outcome')
        plt.ylabel(feature)
        plt.legend(title='Condition Met', labels=['No', 'Yes'])
        plt.show()

def visualize_condition_probability(df, rule, outcome_col):
    for condition in rule['conditions']:
        feature, threshold, inequality = condition['feature'], condition['threshold'], condition['inequality']
        meets_condition = (df[feature] > threshold) if inequality == '>' else (df[feature] <= threshold)
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df.loc[meets_condition, outcome_col], label='Meets Condition', fill=True, color="blue")
        sns.kdeplot(df.loc[~meets_condition, outcome_col], label='Does Not Meet Condition', fill=True, color="orange")
        plt.title(f'Effect of {feature} {inequality} {threshold} on Outcome')
        plt.xlabel('Outcome')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def visualize_condition_effect(df, rule):
    for condition in rule['conditions']:
        plt.figure(figsize=(10, 6))
        feature, threshold, inequality = condition['feature'], condition['threshold'], condition['inequality']
        mask = df[feature] > threshold if inequality == '>' else df[feature] <= threshold
        sns.histplot(df[feature], color="grey", label='All Data', kde=True)
        sns.histplot(df.loc[mask, feature], color="blue", label='Condition Met', kde=True)
        plt.title(f'Effect of {feature} {inequality} {threshold}')
        plt.xlabel(f'{feature} values')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def visualize_feature_hexbin(df, feature_x, feature_y='loan_amnt_n', outcome_col='Outcome'):
    plt.figure(figsize=(10, 8))
    plt.hexbin(df[feature_x], df[feature_y], C=df[outcome_col], reduce_C_function=np.mean, gridsize=50, cmap='viridis')
    plt.colorbar(label=f'Average {outcome_col}')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'Hexbin plot of {feature_y} vs {feature_x} colored by average {outcome_col}')
    plt.grid(True)
    plt.show()

def visualize_feature_hexbin_with_thresholds(df, feature_x, feature_y='loan_amnt_n', outcome_col='Outcome', conditions=None):
    plt.figure(figsize=(10, 8))
    plt.hexbin(df[feature_x], df[feature_y], C=df[outcome_col], reduce_C_function=np.mean, gridsize=50, cmap='viridis')
    cbar = plt.colorbar(label=f'Average {outcome_col}')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'Hexbin plot of {feature_y} vs {feature_x} colored by average {outcome_col}')

    # Draw lines for thresholds
    if conditions:
        for condition in conditions:
            if condition['feature'] == feature_x:
                threshold = condition['threshold']
                plt.axvline(x=threshold, color='red', linestyle='--', label=f"{condition['feature']} {condition['inequality']} {threshold:.1f}")

    # Avoiding duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True)
    plt.show()


def visualize_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()



def plot_feature_importance(model, feature_names):
    """
    Creates a bar chart of the feature importances from a decision tree model.

    Parameters:
    - model: The trained decision tree model.
    - feature_names: List of feature names used in the model.

    """
    importances = model.feature_importances_
    indices = sorted(range(len(importances)), key=lambda k: importances[k], reverse=True)

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(indices)])
    plt.tight_layout()
    plt.show()

def visualize_rule_effect(df, rule, outcome_col):
    """
    Visualizes the effect of a single rule on the outcome by plotting the distribution of the outcome for data points
    that satisfy the rule compared to those that do not.

    Parameters:
    - df: DataFrame, the dataset including the features and the outcome.
    - rule: dict, a single rule, which is a dictionary containing the conditions and the outcome.
    - outcome_col: str, the name of the outcome column in the dataset.
    
    """
    # Apply rule conditions to filter the dataset
    rule_df = df.copy()
    for condition in rule['conditions']:
        feature, inequality, threshold = condition['feature'], condition['inequality'], condition['threshold']
        if inequality == "<=":
            rule_df = rule_df[rule_df[feature] <= threshold]
        else:
            rule_df = rule_df[rule_df[feature] > threshold]

    # Visualize the distribution of outcomes for data points satisfying the rule
    plt.figure(figsize=(12, 8))
    sns.countplot(x=outcome_col, data=rule_df)
    plt.title(f"Distribution of '{outcome_col}' for data points satisfying the rule")
    plt.xlabel(outcome_col)
    plt.ylabel('Count')
    plt.show()

def generate_density_plot(df, x_feature, y_feature, hue):
    """
    Generates a density plot for given features in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - x_feature: str, the name of the x-axis feature for the density plot.
    - y_feature: str, the name of the y-axis feature for the density plot.
    - hue: str, the name of the categorical variable that colors the plot according to its values.

    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=x_feature, y=y_feature, hue=hue, fill=True)
    plt.title('Density Plot')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.show()

def visualize_rules(df, rules):
    """
    Visualizes the rules on the dataset.
    
    Parameters:
    - df: DataFrame containing the data.
    - rules: List of rule dictionaries.
    """
    for rule in rules:
        # Create a new figure for each rule
        plt.figure(figsize=(12, 8))
        filtered_df = df.copy()
        
        # Apply each condition in the rule to filter the data
        for condition in rule['conditions']:
            if condition['inequality'] == '<=':
                filtered_df = filtered_df[filtered_df[condition['feature']] <= condition['threshold']]
            else:
                filtered_df = filtered_df[filtered_df[condition['feature']] > condition['threshold']]

        # Plot the distribution of the feature for both filtered and overall data to show the rule's impact
        sns.kdeplot(data=df, x=condition['feature'], label='All Data', fill=True)
        sns.kdeplot(data=filtered_df, x=condition['feature'], label='Filtered by Rule', fill=True)
        
        plt.title(f"Rule Impact: Outcome = {rule['outcome']} with Confidence = {rule['confidence']:.2f} and Support = {rule['support']}")
        plt.legend()
        plt.xlabel(condition['feature'])
        plt.ylabel('Density')
        plt.show()

def visualize_rule_with_subplots(df, rule, outcome_col):
    """
    Visualizes each condition of the rule in a separate subplot, comparing the outcome distributions.

    Parameters:
    - df: DataFrame containing the data.
    - rule: A dictionary representing the rule to visualize.
    - outcome_col: str, the column name of the outcome variable.
    """
    num_conditions = len(rule['conditions'])
    fig, axs = plt.subplots(num_conditions, 1, figsize=(10, 5 * num_conditions), constrained_layout=True)

    if num_conditions == 1:
        axs = [axs]

    for idx, condition in enumerate(rule['conditions']):
        # Apply the current condition to filter the data
        if condition['inequality'] == '<=':
            condition_df = df[df[condition['feature']] <= condition['threshold']]
        else:
            condition_df = df[df[condition['feature']] > condition['threshold']]

        # Plot the outcome distribution for both filtered and overall data
        sns.histplot(data=df, x=outcome_col, ax=axs[idx], kde=True, stat="density", label='All Data', color="blue", alpha=0.5)
        sns.histplot(data=condition_df, x=outcome_col, ax=axs[idx], kde=True, stat="density", label='Filtered by Condition', color="orange", alpha=0.5)
        axs[idx].set_title(f"Condition {idx + 1}: {condition['feature']} {condition['inequality']} {condition['threshold']}")
        axs[idx].legend()

    plt.show()



def visualize_confusion_matrix(df, rule, outcome_col):
    """
    Applies a rule to the DataFrame, predicts the outcome, and visualizes the confusion matrix.
    
    Parameters:
    - df: DataFrame containing the data.
    - rule: Dictionary representing the rule to visualize.
    - outcome_col: String, the name of the outcome column.
    """
    filtered_df = df.copy()
    for condition in rule['conditions']:
        if condition['inequality'] == '<=':
            filtered_df = filtered_df[filtered_df[condition['feature']] <= condition['threshold']]
        else:
            filtered_df = filtered_df[filtered_df[condition['feature']] > condition['threshold']]
    
    # Assuming a binary outcome where the rule predicts a single class
    predicted_outcome = rule['outcome']
    actual_outcomes = filtered_df[outcome_col]
    predicted_outcomes = np.full_like(actual_outcomes, fill_value=predicted_outcome)

    # Generate and display the confusion matrix
    cm = confusion_matrix(actual_outcomes, predicted_outcomes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(df[outcome_col]))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Rule')
    plt.show()

def visualize_condition_impact_on_outcome(df, rule, outcome_col):
    """
    Visualizes the impact of each condition in a rule on the outcome.

    Parameters:
    - df: DataFrame containing the data.
    - rule: Dictionary representing a rule with conditions.
    - outcome_col: String, the column name of the outcome variable.
    """
    for condition in rule['conditions']:
        feature, threshold, inequality = condition['feature'], condition['threshold'], condition['inequality']
        condition_met = df[feature] > threshold if inequality == '>' else df[feature] <= threshold
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=df[outcome_col], y=df[feature], hue=condition_met, split=True, inner='quartile')
        plt.title(f"Impact of {feature} {inequality} {threshold} on {outcome_col}")
        plt.xlabel('Outcome')
        plt.ylabel(feature)
        plt.legend(title='Condition Met', labels=['No', 'Yes'])
        plt.show()


def visualize_condition_probability(df, rule, outcome_col):
    """
    Visualizes the probability of the outcome for data points that meet each condition compared to those that don't.
    
    Parameters:
    - df: DataFrame containing the data.
    - rule: Dictionary representing a rule with conditions.
    - outcome_col: String, the name of the outcome column.
    """
    for condition in rule['conditions']:
        feature, threshold, inequality = condition['feature'], condition['threshold'], condition['inequality']
        meets_condition = (df[feature] > threshold) if inequality == '>' else (df[feature] <= threshold)
        
        # Create a new figure for each condition
        plt.figure(figsize=(10, 6))
        sns.kdeplot(df.loc[meets_condition, outcome_col], label='Meets Condition', fill=True)
        sns.kdeplot(df.loc[~meets_condition, outcome_col], label='Does Not Meet Condition', fill=True)
        plt.title(f'Effect of {feature} {inequality} {threshold} on Outcome')
        plt.xlabel('Outcome')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

def visualize_condition_effect(df, rule):
    """
    Visualizes the effect of each condition in the rule on the distribution of the feature.

    Parameters:
    - df: DataFrame containing the data.
    - rule: Dictionary representing the rule, with conditions and expected outcome.
    """
    for condition in rule['conditions']:
        plt.figure(figsize=(10, 6))
        feature, threshold, inequality = condition['feature'], condition['threshold'], condition['inequality']
        mask = df[feature] > threshold if inequality == '>' else df[feature] <= threshold
        sns.histplot(df[feature], color="grey", label='All Data', kde=True)
        sns.histplot(df.loc[mask, feature], color="blue", label='Condition Met', kde=True)
        plt.title(f'Effect of {feature} {inequality} {threshold}')
        plt.xlabel(f'{feature} values')
        plt.ylabel('Density')
        plt.legend()
        plt.show()


def visualize_feature_hexbin(df, feature_x, feature_y='loan_amnt_n', outcome_col='Outcome'):
    """
    Generates a hexbin plot to visualize the relationship between two features and their impact on the outcome.

    Parameters:
    - df: DataFrame containing the data.
    - feature_x: String, the name of the feature to display on the x-axis.
    - feature_y: String, default 'loan_amnt_n', the name of the feature to display on the y-axis.
    - outcome_col: String, the name of the outcome column to color-code the hexbins.
    """
    plt.figure(figsize=(10, 8))
    plt.hexbin(df[feature_x], df[feature_y], C=df[outcome_col], reduce_C_function=np.mean, gridsize=50, cmap='viridis')
    plt.colorbar(label=f'Average {outcome_col}')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'Hexbin plot of {feature_y} vs {feature_x} colored by average {outcome_col}')
    plt.grid(True)
    plt.show()




def visualize_feature_importance(model, feature_names):
    """
    Creates a bar chart of the feature importances from a decision tree model.

    Parameters:
    - model: The trained decision tree model.
    - feature_names: List of feature names used in the model, excluding the target.

    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]  # Sort the feature indices by importance

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], color='skyblue', align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

