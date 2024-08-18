#rule_extraction.py:
def extract_rules(tree, feature_names):
    """
    Extracts human-readable decision rules from a decision tree model's structure,
    along with their support and confidence.

    Parameters:
    - tree: The decision tree model's tree_ attribute.
    - feature_names: List of feature names used in the model.

    Returns:
    - rules: A list of dictionaries, each representing a decision rule with its support and confidence.
    """
    def recurse(node, path_conditions, current_rule_support):
        if tree.children_left[node] != tree.children_right[node]:  # not a leaf node
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            left_child = tree.children_left[node]
            right_child = tree.children_right[node]

            # Visit left
            left_conditions = path_conditions + [{'feature': name, 'threshold': threshold, 'inequality': '<='}]
            left_support = tree.n_node_samples[left_child]
            recurse(left_child, left_conditions, left_support)

            # Visit right
            right_conditions = path_conditions + [{'feature': name, 'threshold': threshold, 'inequality': '>'}]
            right_support = tree.n_node_samples[right_child]
            recurse(right_child, right_conditions, right_support)
        else:
            # Leaf node
            outcome = tree.value[node][0].argmax()
            confidence = tree.value[node][0][outcome] / sum(tree.value[node][0])
            rules_list.append({"conditions": path_conditions, "outcome": outcome, "support": current_rule_support, "confidence": confidence})

    rules_list = []
    recurse(0, [], tree.n_node_samples[0])
    return rules_list

def simplify_rules(rules):
    """
    Simplifies the extracted rules by sorting them based on their predictive power.

    Parameters:
    - rules: A list of dictionaries, each representing a decision rule with its support and confidence.

    Returns:
    - simplified_rules: A sorted list of simplified rules by confidence and support.
    """
    simplified_rules = []
    rule_dict = {}

    # Combine rules with the same conditions
    for rule in rules:
        # Create a frozenset of conditions to uniquely identify rules
        condition_set = frozenset((cond['feature'], cond['threshold'], cond['inequality']) for cond in rule['conditions'])
        
        # If a rule with the same conditions already exists, compare and store the one with higher confidence
        if condition_set not in rule_dict or rule['confidence'] > rule_dict[condition_set]['confidence']:
            rule_dict[condition_set] = rule
        elif rule['confidence'] == rule_dict[condition_set]['confidence']:
            # If confidences are equal, use the rule with the higher support
            if rule['support'] > rule_dict[condition_set]['support']:
                rule_dict[condition_set] = rule

    # Sort the rules by confidence and support
    simplified_rules = sorted(rule_dict.values(), key=lambda r: (r['confidence'], r['support']), reverse=True)

    return simplified_rules