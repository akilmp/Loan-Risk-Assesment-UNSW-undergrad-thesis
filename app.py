import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from preprocessing import clean_data, encode_features, split_features_target
from rule_extraction import extract_rules, simplify_rules
from sklearn.ensemble import GradientBoostingClassifier


# Global variable to store rules in RDR format
rdr_rules = []
# Global variable to store newly created rules
new_rules = []


# Cache expensive functions
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)
    df = encode_features(df)
    return df

@st.cache_data
def train_model(X, y, model_type, max_depth, min_samples_split, min_samples_leaf, n_estimators=None, learning_rate=0.1, subsample=1.0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'Decision Tree':
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=42
        )
    
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

@st.cache_data
def cross_validate_model(_model, X, y):
    cv_scores = cross_val_score(_model, X, y, cv=5, n_jobs=-1)  # Use all available cores for cross-validation
    return np.mean(cv_scores), np.std(cv_scores)



# Main function to run the entire Streamlit application
def main():
    st.set_page_config(page_title="Loan Risk Assessment Dashboard", layout="wide")
    st.title("Loan Risk Assessment Dashboard")

    # Initialize session state variables
    if 'rule_index' not in st.session_state:
        st.session_state.rule_index = 1

    if 'selected_rule_index' not in st.session_state:
        st.session_state.selected_rule_index = 1

    # Ensure conditions and exceptions persist when model type or pruning parameters are changed
    if 'selected_conditions' not in st.session_state:
        st.session_state.selected_conditions = []

    if 'selected_exceptions' not in st.session_state:
        st.session_state.selected_exceptions = []

    # Initialize pruning parameters if not already set
    if 'max_depth' not in st.session_state:
        st.session_state.max_depth = 5
    
    if 'min_samples_split' not in st.session_state:
        st.session_state.min_samples_split = 50

    if 'min_samples_leaf' not in st.session_state:
        st.session_state.min_samples_leaf = 25

    if 'n_estimators' not in st.session_state:
        st.session_state.n_estimators = 50

    # Sidebar for navigation and global settings
    with st.sidebar:
        st.header("Settings")
        model_type = st.selectbox("Select Model Type", ["Decision Tree", "Random Forest", "Gradient Boosting"])
        st.session_state.feature_threshold = st.slider("Set Feature Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.0)
        
        # Directly link the number input to st.session_state.selected_rule_index
        st.number_input(
            "Enter rule number to graph:",
            min_value=1,
            step=1,
            value=st.session_state.selected_rule_index,
            key="selected_rule_index",
        )

        st.subheader("Pruning Parameters")
        st.slider(
            "Set Max Depth", 
            min_value=1, 
            max_value=20, 
            step=1, 
            value=st.session_state.max_depth, 
            key="max_depth"
        )
        st.slider(
            "Set Min Samples Split", 
            min_value=2, 
            max_value=100, 
            step=1, 
            value=st.session_state.min_samples_split, 
            key="min_samples_split"
        )
        st.slider(
            "Set Min Samples Leaf", 
            min_value=1, 
            max_value=100, 
            step=1, 
            value=st.session_state.min_samples_leaf, 
            key="min_samples_leaf"
        )

        if model_type in ["Random Forest", "Gradient Boosting"]:
            st.slider(
                "Set Number of Estimators", 
                min_value=10, 
                max_value=200, 
                step=10, 
                value=st.session_state.n_estimators, 
                key="n_estimators"
            )
            if model_type == "Gradient Boosting":
                st.slider(
                    "Set Learning Rate", 
                    min_value=0.01, 
                    max_value=1.0, 
                    step=0.01, 
                    value=0.1, 
                    key="learning_rate"
                )
                st.slider(
                    "Set Subsample", 
                    min_value=0.1, 
                    max_value=1.0, 
                    step=0.1, 
                    value=1.0, 
                    key="subsample"
                )


    # Step 1: Upload Dataset
    with st.expander("Step 1: Upload Dataset"):
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = load_and_preprocess_data(uploaded_file)  # 
            st.write("Data Preview", df.head())

            X, y = split_features_target(df, 'Outcome')  # 
            
            with st.spinner('Training the model, please wait...'):
                model, X_train, X_test, y_train, y_test = train_model(  # 
                    X, y, model_type, st.session_state.max_depth, st.session_state.min_samples_split,
                    st.session_state.min_samples_leaf, st.session_state.n_estimators
                )

            mean_cv_score, std_cv_score = cross_validate_model(model, X, y)  # 
            st.write(f"Cross-Validation Accuracy: {mean_cv_score:.2%} ± {std_cv_score:.2%}")

            if model_type == 'Decision Tree':
                rules = extract_rules(model.tree_, X.columns)  # 
            elif model_type == 'Random Forest':
                rules = extract_rules(model.estimators_[0].tree_, X.columns)  # 
            if model_type == 'Gradient Boosting':
                rules = extract_rules(model.estimators_[0][0].tree_, X.columns)

                
            simplified_rules = simplify_rules(rules)  # 

            global rdr_rules
            rdr_rules = simplified_rules

    if uploaded_file is not None:
        # Step 2: Display Rules as Table
        with st.expander("Step 2: Display Rules as Table"):
            display_rules_as_table(rdr_rules, df)  # 

        # Step 3: Graph a Rule with Feature Threshold
        with st.expander("Step 3: Graph a Rule with Feature Threshold"):
            if 'rdr_rules' in globals() and rdr_rules:
                selected_rule = rdr_rules[st.session_state.selected_rule_index - 1]
                features_of_interest = [cond['feature'] for cond in selected_rule['conditions'] if model.feature_importances_[list(X.columns).index(cond['feature'])] > st.session_state.feature_threshold]

                if features_of_interest:
                    selected_feature = st.selectbox("Select feature to graph", features_of_interest)
                    conditions = [cond for cond in selected_rule['conditions'] if cond['feature'] == selected_feature]
                    fig = visualize_feature_hexbin_with_thresholds(df, selected_feature, conditions=conditions)  # 
                    st.plotly_chart(fig)
                else:
                    st.write("No features meet the threshold criteria.")

        # Step 4: Display Feature Importance
        with st.expander("Step 4: Display Feature Importance"):
            fig = visualize_feature_importance(model, df.columns[:-1])  # 
            st.plotly_chart(fig)

        # Step 5: Create New Rule
        with st.expander("Step 5: Create New Rule"):
            create_new_rule(df, rdr_rules)  # Use the updated `create_new_rule` function



def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(
        max_depth=st.session_state.max_depth, 
        min_samples_split=st.session_state.min_samples_split, 
        min_samples_leaf=st.session_state.min_samples_leaf, 
        random_state=42
    )
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    st.write(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2%} ± {np.std(cv_scores):.2%}")

    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))

def display_rules_as_table(rules, df):
    table = []
    for idx, rule in enumerate(rules):
        conditions = ', '.join([f"{cond['feature']} {cond['inequality']} {cond['threshold']}" for cond in rule['conditions']])
        exceptions = ', '.join([f"{exc['feature']} {exc['inequality']} {exc['threshold']}" for exc in rule.get('exceptions', [])])

        # Calculate confidence and support using the unified function
        confidence, support = calculate_confidence_and_support(df, rule)
        
        # Only add rows that have meaningful data
        if conditions or exceptions or rule['outcome'] or confidence or support:
            table.append([idx + 1, conditions, exceptions, rule['outcome'], confidence, support])

    headers = ["Rule Number", "Conditions", "Exceptions", "Outcome", "Confidence", "Support"]
    df_table = pd.DataFrame(table, columns=headers)

    # Drop rows where all values are NaN or empty
    df_table = df_table.dropna(how='all')

    # Also drop rows where all relevant columns (Conditions, Exceptions, Outcome, Confidence, Support) are empty
    df_table = df_table[
        (df_table['Conditions'] != '') |
        (df_table['Exceptions'] != '') |
        (df_table['Outcome'].notna()) |
        (df_table['Confidence'].notna()) |
        (df_table['Support'].notna())
    ].reset_index(drop=True)

    # Correctly specify table styles using a list of dictionaries
    df_styled = df_table.style.set_table_styles(
        [
            {'selector': 'th', 'props': [('text-align', 'center')]},  # Center-align header
            {'selector': 'td', 'props': [('text-align', 'center')]},  # Center-align all cells
            {'selector': 'td:nth-child(2)', 'props': [('text-align', 'left'), ('width', '50%')]},  # Left-align "Conditions" column
        ]
    ).format(
        {
            'Confidence': '{:.2%}',
            'Support': '{:,}'  # Add thousand separators for support
        }).background_gradient(
        subset=['Confidence'], cmap='YlGn'  # Gradient coloring for Confidence
    ).background_gradient(
        subset=['Support'], cmap='Blues'  # Gradient coloring for Support
    ).set_properties(
        **{'white-space': 'pre-wrap'}  # Wrap long text
    )

    # Display the styled DataFrame
    st.dataframe(df_styled, height=400, width=1000)  # Adjust width and height to fit content



def visualize_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig = px.bar(
        x=[feature_names[i] for i in indices],
        y=importances[indices],
        labels={'x': 'Feature', 'y': 'Importance'},
        title='Feature Importances',
        color=importances[indices],
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_layout(
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    )
    return fig

def visualize_feature_hexbin_with_thresholds(df, feature_x, feature_y='loan_amnt_n', outcome_col='Outcome', conditions=None):
    # Custom color scale to distinguish zero values
    custom_colorscale = [
        [0, 'white'],      # Background color
        [0.0001, 'blue'],  # Color for very low values close to zero
        [0.5, 'green'],    # Mid-range color
        [1, 'red']         # High value color
    ]

    # Creating the density heatmap with the custom color scale
    fig = px.density_heatmap(
        df, x=feature_x, y=feature_y, z=outcome_col,
        histfunc='avg', nbinsx=50, nbinsy=50,
        labels={feature_x: feature_x, feature_y: feature_y, outcome_col: f'Average {outcome_col}'},
        color_continuous_scale=custom_colorscale,
        range_color=[0.0001, df[outcome_col].max()]  # Ensure zero is not included in the color range
    )
    fig.update_layout(
        title=f'Hexbin plot of {feature_y} vs {feature_x} colored by average {outcome_col}',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgrey'),
        coloraxis_colorbar=dict(
            title='Average Outcome',
            tickvals=[0.0001, 0.5, 1],
            ticktext=['0', '0.5', '1']
        )
    )

    if conditions:
        for condition in conditions:
            if condition['feature'] == feature_x:
                threshold = condition['threshold']
                fig.add_vline(
                    x=threshold,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"{condition['inequality']} {threshold}",
                    annotation_position="top left",
                    annotation_font_color="black"  # Change annotation font color to black
                )

    return fig

def calculate_confidence_and_support(df, rule):
    filtered_df = df.copy()
    
    # Apply each condition to filter the dataset
    for condition in rule['conditions']:
        if condition['inequality'] == '<=':
            filtered_df = filtered_df[filtered_df[condition['feature']] <= condition['threshold']]
        else:
            filtered_df = filtered_df[filtered_df[condition['feature']] > condition['threshold']]

    # Apply each exception to further filter out the dataset
    for exception in rule.get('exceptions', []):
        if exception['inequality'] == '<=':
            filtered_df = filtered_df[filtered_df[exception['feature']] > exception['threshold']]
        else:
            filtered_df = filtered_df[filtered_df[exception['feature']] <= exception['threshold']]

    # Support is the count of rows that satisfy all conditions and are not excluded by exceptions
    support = len(filtered_df)

    # Confidence is the proportion of those rows that have the desired outcome
    if support > 0:
        confidence = filtered_df['Outcome'].value_counts(normalize=True).get(rule['outcome'], 0)
    else:
        confidence = 0.0

    return confidence, support

def format_condition(cond):
    return f"{cond['feature']} {cond['inequality']} {cond['threshold']}"

def update_rule_index():
    pass  # You can remove this if no additional logic is needed here

# Function to create new rules
def create_new_rule(df, rules):
    st.write("Create New Rule")

    # Ensure selected conditions and exceptions persist in session state
    if 'selected_conditions' not in st.session_state:
        st.session_state.selected_conditions = []
    
    if 'selected_exceptions' not in st.session_state:
        st.session_state.selected_exceptions = []
    
    # This part will initialize the session state if it does not exist
    if 'selected_rule_index' not in st.session_state:
        st.session_state.selected_rule_index = 1  # Default to 1 if not set

    # Retrieve the current rule index from the session state
    selected_rule = rules[st.session_state.selected_rule_index - 1]

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Add conditions from existing rules", "Create your own new condition"])

    with tab1:
        readable_conditions = [format_condition(cond) for cond in selected_rule['conditions']]
        new_condition = st.selectbox("Select a condition to add from existing rules:", [""] + readable_conditions)
        
        if new_condition and new_condition not in st.session_state.selected_conditions:
            st.session_state.selected_conditions.append(new_condition)

    with tab2:
        cols = st.columns(3)
        with cols[0]:
            feature = st.selectbox("Select a feature", df.columns, key="custom_condition_feature")
        with cols[1]:
            inequality = st.selectbox("Select an inequality", ["<=", ">"], key="custom_condition_inequality")
        with cols[2]:
            threshold = st.number_input(
                f"Set a threshold for {feature}:",
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                key="custom_condition_threshold"
            )

        if st.button("Add Custom Condition"):
            custom_condition = f"{feature} {inequality} {threshold}"
            st.session_state.selected_conditions.append(custom_condition)

    # Display currently selected conditions with remove buttons
    st.write("### Currently selected conditions:")
    conditions_to_remove = []
    for condition in st.session_state.selected_conditions:
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.write(condition)
        with col2:
            if st.button(f"❌", key=f"remove_condition_{condition}"):
                conditions_to_remove.append(condition)

    # Remove selected conditions
    for condition in conditions_to_remove:
        st.session_state.selected_conditions.remove(condition)

    # Constructing the new_rule_conditions from selected conditions
    new_rule_conditions = []
    for condition in st.session_state.selected_conditions:
        if condition in readable_conditions:
            new_rule_conditions.append(selected_rule['conditions'][readable_conditions.index(condition)])
        else:
            feature, inequality, threshold = condition.split()
            threshold = float(threshold)
            new_rule_conditions.append({'feature': feature, 'inequality': inequality, 'threshold': threshold})

    # Section for handling exceptions
    st.write("### Handle Exceptions:")
    exception_tab1, exception_tab2 = st.tabs(["Add conditions or exceptions from selected rule", "Create your own exception"])

    with exception_tab1:
        # Combine conditions and exceptions from the selected rule
        readable_exceptions_and_conditions = [format_condition(cond) for cond in selected_rule['conditions']] + \
                                             [format_condition(exc) for exc in selected_rule.get('exceptions', [])]

        if not readable_exceptions_and_conditions:
            st.warning("No conditions or exceptions available in the selected rule.")
        else:
            selected_exception = st.selectbox(
                "Select a condition or exception to add from the selected rule:", 
                [""] + readable_exceptions_and_conditions,
                key="exception_rule_selectbox"
            )
        
            if selected_exception and selected_exception not in st.session_state.selected_exceptions:
                st.session_state.selected_exceptions.append(selected_exception)

    with exception_tab2:
        cols = st.columns(3)
        with cols[0]:
            exception_feature = st.selectbox("Select a feature for exception", df.columns, key="custom_exception_feature")
        with cols[1]:
            exception_inequality = st.selectbox("Select an inequality for exception", ["<=", ">"], key="custom_exception_inequality")
        with cols[2]:
            exception_threshold = st.number_input(
                f"Set a threshold for {exception_feature}:",
                min_value=float(df[exception_feature].min()),
                max_value=float(df[exception_feature].max()),
                key="custom_exception_threshold"
            )

        if st.button("Add Custom Exception", key="add_custom_exception"):
            custom_exception = f"{exception_feature} {exception_inequality} {exception_threshold}"
            st.session_state.selected_exceptions.append(custom_exception)

    # Display currently selected exceptions with remove buttons
    st.write("### Currently selected exceptions:")
    exceptions_to_remove = []
    for exception in st.session_state.selected_exceptions:
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.write(exception)
        with col2:
            if st.button(f"❌", key=f"remove_exception_{exception}"):
                exceptions_to_remove.append(exception)

    # Remove selected exceptions
    for exception in exceptions_to_remove:
        st.session_state.selected_exceptions.remove(exception)

    # Constructing the new_rule_exceptions from selected exceptions
    new_rule_exceptions = []
    for exception in st.session_state.selected_exceptions:
        if exception in readable_exceptions_and_conditions:
            new_rule_exceptions.append({
                'feature': exception.split()[0],
                'inequality': exception.split()[1],
                'threshold': float(exception.split()[2])
            })
        else:
            feature, inequality, threshold = exception.split()
            threshold = float(threshold)
            new_rule_exceptions.append({'feature': feature, 'inequality': inequality, 'threshold': threshold})

    new_rule_outcome = st.selectbox("Outcome:", df['Outcome'].unique(), key="new_rule_outcome")
    
    new_rule = {
        "conditions": new_rule_conditions, 
        "exceptions": new_rule_exceptions, 
        "outcome": new_rule_outcome
    }
    
    confidence, support = calculate_confidence_and_support(df, new_rule)  # 
    st.write(f"Calculated Confidence: {confidence:.2%}")
    st.write(f"Calculated Support: {support} instances")
    
    if st.button("Create Rule", key="create_rule_button"):
        global new_rules  # Reference the global variable
        new_rules.append(new_rule)
        st.write("New rule created successfully.")
        
        # Display all new rules, including the newly created one
        st.write("### All Created Rules")
        display_rules_as_table(new_rules, df)  # 

        st.session_state.selected_conditions = []  # Reset conditions for the next rule
        st.session_state.selected_exceptions = []  # Reset exceptions for the next rule

if __name__ == "__main__":
    main()