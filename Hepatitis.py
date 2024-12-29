#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Load and preprocess data
data = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Assuming the last 3 columns are the target variables (diseases)
X = data.drop(columns=['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'])
y = data[['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis']]

# Step 2: Initialize lists to store resampled data
X_resampled_list = []
y_resampled_list = []

# Step 3: Apply SMOTE to each disease column separately
smote = SMOTE(random_state=42)

for i in range(y.shape[1]):
    X_resampled_i, y_resampled_i = smote.fit_resample(X, y.iloc[:, i])
    X_resampled_list.append(X_resampled_i)
    y_resampled_list.append(y_resampled_i)

# Step 4: Split resampled data into training and testing sets
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

for i in range(y.shape[1]):
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_resampled_list[i], y_resampled_list[i], test_size=0.2, random_state=42)
    X_train_list.append(X_train_i)
    X_test_list.append(X_test_i)
    y_train_list.append(y_train_i)
    y_test_list.append(y_test_i)

# Step 5: Initialize Decision Tree Classifier and train for each disease
dt_models = []

for i in range(y.shape[1]):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_list[i], y_train_list[i])
    dt_models.append(dt)
    
    # Step 6: Evaluate the model for each disease
    y_pred_i = dt.predict(X_test_list[i])
    
    print(f"Classification Report for {y.columns[i]}:")
    print(classification_report(y_test_list[i], y_pred_i))
    
    print(f"\nConfusion Matrix for {y.columns[i]}:")
    print(confusion_matrix(y_test_list[i], y_pred_i))
    print("\n")
    
    # Step 7: Visualize Decision Tree
    plt.figure(figsize=(15, 10))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['0', '1'])
    plt.title(f"Decision Tree for {y.columns[i]} Prediction")
    plt.show()

# Step 8: Visualize feature importances for the last trained model (Chronic Hepatitis)
plt.figure(figsize=(10, 6))
importances = pd.Series(dt_models[-1].feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.show()


# In[1]:


#Random Forest Final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import graphviz
from sklearn.tree import export_graphviz

# Define the visualize_tree function
def visualize_tree(model, feature_names, target, tree_index=0):
    """
    Visualizes a single decision tree from a Random Forest model using Graphviz.
    
    Parameters:
    - model: The trained Random Forest model
    - feature_names: List of feature names (column names of the input data)
    - target: Name of the target variable (class label)
    - tree_index: Index of the tree in the Random Forest to visualize (default is 0)
    """
    # Retrieve the decision tree from the ensemble
    estimator = model.estimators_[tree_index]

    # Export the decision tree as a Graphviz DOT format string
    dot_data = export_graphviz(estimator, out_file=None,
                               feature_names=feature_names,
                               class_names=[str(class_name) for class_name in model.classes_],
                               filled=True, rounded=True,
                               special_characters=True)

    # Create a Graphviz Source object from the DOT data
    graph = graphviz.Source(dot_data)

    # Save the rendered tree as a PDF file
    output_file_pdf = f"random_forest_tree_{target.replace(' ', '_')}.pdf"
    graph.render(output_file_pdf, format='pdf')

    # Save the rendered tree as PNG
    output_file_png = f"random_forest_tree_{target.replace(' ', '_')}.png"
    graph.render(output_file_png, format='png')

    # Display the rendered tree
    graph.view()

# Load the data from Excel
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx', sheet_name='Sheet1')  # Adjust sheet_name as necessary

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last three)
y = df.iloc[:, -3:]  # Target (last three columns)

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Models for each Disease
models = {}
for disease in y_train.columns:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train[disease])
    models[disease] = rf

# Step 5: Evaluate the Models and Plot Feature Importances for Each Disease
top_n = 10  # Number of top features to display, adjust as needed

for disease, model in models.items():
    # Evaluate the model
    y_pred = model.predict(X_test)
    
    # Print Classification Report
    print(f"\nClassification Report for Disease {disease}:")
    print(classification_report(y_test[disease], y_pred))
    
    # Print Confusion Matrix
    print(f"\nConfusion Matrix for Disease {disease}:")
    print(confusion_matrix(y_test[disease], y_pred))
    
    # Plot Feature Importances
    importances = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances for Disease {disease}")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"feature_importances_{disease.replace(' ', '_')}.png")
    plt.show()
    
    # Visualize a single tree from the Random Forest
    visualize_tree(model, feature_names, disease, tree_index=0)


# In[3]:


get_ipython().system('pip install graphviz')


# In[38]:


#Logistic Regression Extended
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Assuming your DataFrame is named df and the last three columns are the target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# Apply SMOTE to each target variable separately
smote = SMOTE(random_state=42)

# Helper function to train the model and extract feature importances
def train_and_plot_feature_importance(X, y, disease_name):
    X_res, y_res = smote.fit_resample(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Train Logistic Regression Model
    logreg_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
    logreg_model.fit(X_train, y_train)
    
    # Extract feature importances
    feature_importances = logreg_model.coef_[0]
    
    # Create a DataFrame for visualization
    feature_importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title(f'Feature Importances for {disease_name}')
    plt.gca().invert_yaxis()
    plt.show()
    
# Plot feature importances for each disease
train_and_plot_feature_importance(X, y_viral, 'Viral Hepatitis')
train_and_plot_feature_importance(X, y_acute, 'Acute Hepatitis')
train_and_plot_feature_importance(X, y_chronic, 'Chronic Hepatitis')


# In[6]:


#Hepatitis final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X, y, disease_name):
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_res, y_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'mean_cv_score': mean_cv_score
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_viral = train_and_evaluate(X, y_viral, 'Viral Hepatitis')
results_acute = train_and_evaluate(X, y_acute, 'Acute Hepatitis')
results_chronic = train_and_evaluate(X, y_chronic, 'Chronic Hepatitis')

# Plotting the results
def plot_results(results, disease_name):
    labels = list(results.keys())
    accuracy = [results[model]['accuracy'] for model in labels]
    mean_cv_score = [results[model]['mean_cv_score'] for model in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
    bar2 = ax.bar(x + width/2, mean_cv_score, width, label='Mean CV Score')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Model Comparison for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(bar1, padding=3)
    ax.bar_label(bar2, padding=3)

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Plot results for each disease
plot_results(results_viral, 'Viral Hepatitis')
plot_results(results_acute, 'Acute Hepatitis')
plot_results(results_chronic, 'Chronic Hepatitis')


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data from Excel (replace with your actual file path)
file_path = r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx'
df = pd.read_excel(file_path)

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last four)
y = df.iloc[:, -3:]  # Target (last four columns)

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'XGBoost': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB()
}

# Dictionary to store cross-validation scores
cv_scores = {}

# Train and evaluate models for each disease separately
for disease in y_train.columns:
    print(f"Evaluating models for disease: {disease}\n")
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train[disease])
        
        # Cross-validation
        cv_score = cross_val_score(model, X, y[disease], cv=5, scoring='accuracy').mean()
        cv_scores[model_name] = cv_score

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test[disease], y_pred)

        # Print evaluation metrics
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Mean Cross-Validation Score: {cv_score:.3f}")
        print(f"Classification Report for {model_name} - {disease}:")
        print(classification_report(y_test[disease], y_pred))
        print(f"Confusion Matrix for {model_name} - {disease}:")
        print(confusion_matrix(y_test[disease], y_pred))
        print("\n------------------------------------------\n")

    print(f"================================================================================\n")

# Create a DataFrame for mean cross-validation scores
cv_scores_df = pd.DataFrame(list(cv_scores.items()), columns=['Model', 'Mean CV Score'])

# Plotting
plt.figure(figsize=(8, 10))
sns.barplot(x='Model', y='Mean CV Score', data=cv_scores_df, palette=sns.color_palette("OrRd", len(cv_scores_df)))
plt.title('Mean Cross-Validation Score for Different Models')
plt.xlabel('Model')
plt.ylabel('Mean CV Score')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for clarity
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.tight_layout()
plt.show()



# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the data from Excel (replace with your actual file path)
file_path = r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx'
df = pd.read_excel(file_path)

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last four)
y = df.iloc[:, -3:]  # Target (last four columns)

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'XGBoost': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Gaussian Naive Bayes': GaussianNB()
}

# Dictionary to store cross-validation scores
cv_scores = {}

# Train and evaluate models for each disease separately
for disease in y_train.columns:
    print(f"Evaluating models for disease: {disease}\n")
    for model_name, model in models.items():
        # Train model
        model.fit(X_train, y_train[disease])
        
        # Cross-validation
        cv_score = cross_val_score(model, X, y[disease], cv=5, scoring='accuracy').mean()
        cv_scores[model_name] = cv_score

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test[disease], y_pred)

        # Print evaluation metrics
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Mean Cross-Validation Score: {cv_score:.3f}")
        print(f"Classification Report for {model_name} - {disease}:")
        print(classification_report(y_test[disease], y_pred))
        print(f"Confusion Matrix for {model_name} - {disease}:")
        print(confusion_matrix(y_test[disease], y_pred))
        print("\n------------------------------------------\n")

    print(f"================================================================================\n")

# Create a DataFrame for mean cross-validation scores
cv_scores_df = pd.DataFrame(list(cv_scores.items()), columns=['Model', 'Mean CV Score'])

# Plotting
plt.figure(figsize=(8, 10))
sns.barplot(x='Model', y='Mean CV Score', data=cv_scores_df, palette=sns.color_palette("OrRd", len(cv_scores_df)))
plt.title('Mean Cross-Validation Score for Different Models')
plt.xlabel('Model')
plt.ylabel('Mean CV Score')
plt.ylim(0, 1)  # Limit y-axis from 0 to 1 for clarity
plt.xticks(rotation=45, fontsize=14)  # Rotate x-axis labels for better visibility and increase font size
plt.tight_layout()
plt.show()


# In[7]:


import matplotlib.pyplot as plt

# Define function to plot results
def plot_results(results, disease_name):
    labels = list(results.keys())
    accuracy = [results[model]['accuracy'] for model in labels]
    mean_cv_score = [results[model]['mean_cv_score'] for model in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bar1 = ax.bar(x - width/2, accuracy, width, label='Accuracy')
    bar2 = ax.bar(x + width/2, mean_cv_score, width, label='Mean CV Score')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Model Comparison for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(bar1, padding=3)
    ax.bar_label(bar2, padding=3)

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Plot results for each disease
plot_results(results_viral, 'Viral Hepatitis')
plot_results(results_acute, 'Acute Hepatitis')
plot_results(results_chronic, 'Chronic Hepatitis')


# In[26]:


#Decision tree final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Load and preprocess data
data = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Assuming the last 3 columns are the target variables (diseases)
X = data.drop(columns=['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'])
y = data[['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis']]

# Step 2: Initialize lists to store resampled data
X_resampled_list = []
y_resampled_list = []

# Step 3: Apply SMOTE to each disease column separately
smote = SMOTE(random_state=42)

for i in range(y.shape[1]):
    X_resampled_i, y_resampled_i = smote.fit_resample(X, y.iloc[:, i])
    X_resampled_list.append(X_resampled_i)
    y_resampled_list.append(y_resampled_i)

# Step 4: Split resampled data into training and testing sets
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

for i in range(y.shape[1]):
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_resampled_list[i], y_resampled_list[i], test_size=0.2, random_state=42)
    X_train_list.append(X_train_i)
    X_test_list.append(X_test_i)
    y_train_list.append(y_train_i)
    y_test_list.append(y_test_i)

# Step 5: Initialize Decision Tree Classifier and train for each disease
dt_models = []

for i in range(y.shape[1]):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_list[i], y_train_list[i])
    dt_models.append(dt)
    
    # Step 6: Evaluate the model for each disease
    y_pred_i = dt.predict(X_test_list[i])
    
    print(f"Classification Report for {y.columns[i]}:")
    print(classification_report(y_test_list[i], y_pred_i))
    
    print(f"\nConfusion Matrix for {y.columns[i]}:")
    print(confusion_matrix(y_test_list[i], y_pred_i))
    print("\n")
    
    # Step 7: Visualize Decision Tree
    plt.figure(figsize=(15, 10))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['0', '1'])
    plt.title(f"Decision Tree for {y.columns[i]} Prediction")
    plt.show()

# Step 8: Visualize feature importances for the last trained model (Chronic Hepatitis)
plt.figure(figsize=(10, 6))
importances = pd.Series(dt_models[-1].feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.show()


# In[8]:


#gradient boosting final
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Load the data from Excel
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx', sheet_name='Sheet1')  # Adjust sheet_name as necessary

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last three)
y = df.iloc[:, -3:]  # Target (last three columns)

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train XGBoost Models for each Disease
models = {}
for disease in y_train.columns:
    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)
    xgb_model.fit(X_train, y_train[disease])
    models[disease] = xgb_model

    # Step 5: Evaluate the Model
    y_pred = xgb_model.predict(X_test)
    
    print(f"\nClassification Report for Disease {disease}:")
    print(classification_report(y_test[disease], y_pred))
    
    print(f"\nConfusion Matrix for Disease {disease}:")
    print(confusion_matrix(y_test[disease], y_pred))
    
    # Step 6: Visualize Feature Importances
    plt.figure(figsize=(10, 6))
    plot_importance(xgb_model)
    plt.title(f"Feature Importance for Disease {disease}")
    plt.show()

    # Step 7: Visualize an Individual Tree (e.g., the first tree in the ensemble)
    plt.figure(figsize=(15, 10))
    plot_tree(xgb_model, num_trees=0, rankdir='LR')  # num_trees=0 means the first tree
    plt.title(f"XGBoost Tree for Disease {disease}")
    plt.show()


# In[10]:


##final logistic regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.inspection import plot_partial_dependence

# Load the data from Excel
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx', sheet_name='Sheet1')  # Adjust sheet_name as necessary

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last three)
y = df.iloc[:, -3:]  # Target (last three columns)

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression models separately for each disease
models = {}
for disease in y_train.columns:
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train[disease])
    models[disease] = model

    # 1. Plot feature importance
    coefficients = model.coef_[0]
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coefficients)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance in Logistic Regression for {disease}')
    plt.show()

    # 2. Predict probabilities and plot ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test[disease], y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test[disease], y_pred_proba):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Logistic Regression - {disease}')
    plt.legend()
    plt.show()

    # 3. Plot partial dependence for a specific feature (example for the first feature)
    plt.figure(figsize=(10, 6))
    plot_partial_dependence(model, X_train, features=[0], feature_names=X.columns)
    plt.xlabel('Feature Value')
    plt.ylabel('Partial Dependence')
    plt.title(f'Partial Dependence Plot for Feature 1 - {disease}')
    plt.show()

# Additional: Classification report and confusion matrix
for disease, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for Disease {disease}:")
    print(classification_report(y_test[disease], y_pred))

    print(f"\nConfusion Matrix for Disease {disease}:")
    print(confusion_matrix(y_test[disease], y_pred))


# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

# Load the data from Excel
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx', sheet_name='Sheet1')  # Adjust sheet_name as necessary

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last three)
y = df.iloc[:, -3:]  # Target (last three columns)

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train XGBoost Models for each Disease
models = {}
for disease in y_train.columns:
    xgb_model = XGBClassifier(objective='binary:logistic', random_state=42)
    xgb_model.fit(X_train, y_train[disease])
    models[disease] = xgb_model

    # Step 5: Evaluate the Model
    y_pred = xgb_model.predict(X_test)
    
    print(f"\nClassification Report for Disease {disease}:")
    print(classification_report(y_test[disease], y_pred))
    
    print(f"\nConfusion Matrix for Disease {disease}:")
    cm = confusion_matrix(y_test[disease], y_pred)
    print(cm)
    
    # Plot the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"Confusion Matrix for Disease {disease}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Step 6: Visualize Feature Importances
    plt.figure(figsize=(10, 6))
    plot_importance(xgb_model)
    plt.title(f"Feature Importance for Disease {disease}")
    plt.show()

    # Step 7: Visualize an Individual Tree (e.g., the first tree in the ensemble)
    plt.figure(figsize=(15, 10))
    plot_tree(xgb_model, num_trees=0, rankdir='LR')  # num_trees=0 means the first tree
    plt.title(f"XGBoost Tree for Disease {disease}")
    plt.show()


# In[1]:


#Logistic extended
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

# Load the data from Excel
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx', sheet_name='Sheet1')  # Adjust sheet_name as necessary

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last three)
y = df.iloc[:, -3:]  # Target (last three columns)

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression models separately for each disease
models = {}
for disease in y_train.columns:
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train[disease])
    models[disease] = model

    # Predict probabilities and plot ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test[disease], y_pred_proba)
    auc_score = roc_auc_score(y_test[disease], y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Logistic Regression - {disease}')
    plt.legend()
    plt.show()

    # Print Classification Report
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for Disease {disease}:")
    print(classification_report(y_test[disease], y_pred))

    # Print Confusion Matrix
    print(f"\nConfusion Matrix for Disease {disease}:")
    print(confusion_matrix(y_test[disease], y_pred))


# In[7]:


#Gaussian Naive Bayes Final extended
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import numpy as np

# Load the data from Excel
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx', sheet_name='Sheet1')  # Adjust sheet_name as necessary

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last three)
y = df.iloc[:, -3:]  # Target (last three columns)

# Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes models separately for each disease
models = {}
for disease in y_train.columns:
    model = GaussianNB()
    model.fit(X_train, y_train[disease])
    models[disease] = model

    # Predict probabilities and plot ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test[disease], y_pred_proba)
    auc_score = roc_auc_score(y_test[disease], y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Gaussian Naive Bayes - {disease}')
    plt.legend()
    plt.show()

    # Print Classification Report
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for Disease {disease}:")
    print(classification_report(y_test[disease], y_pred))

    # Print Confusion Matrix
    print(f"\nConfusion Matrix for Disease {disease}:")
    print(confusion_matrix(y_test[disease], y_pred))

    # Feature Importance based on mean value per class (interpretatively)
    means_per_class = model.theta_
    feature_importance = np.mean(means_per_class, axis=0)
    sorted_idx = np.argsort(feature_importance)[::-1]

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns[sorted_idx], feature_importance[sorted_idx], align='center')
    plt.xlabel('Mean Feature Value')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance - Gaussian Naive Bayes {disease}')
    plt.show()


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import graphviz
from sklearn.tree import export_graphviz
from sklearn.inspection import plot_partial_dependence

# Define the visualize_tree function
def visualize_tree(model, feature_names, target, tree_index=0):
    estimator = model.estimators_[tree_index]
    dot_data = export_graphviz(estimator, out_file=None,
                               feature_names=feature_names,
                               class_names=[str(class_name) for class_name in model.classes_],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    output_file_pdf = f"random_forest_tree_{target.replace(' ', '_')}.pdf"
    graph.render(output_file_pdf, format='pdf')
    output_file_png = f"random_forest_tree_{target.replace(' ', '_')}.png"
    graph.render(output_file_png, format='png')
    graph.view()

# Load the data from Excel
excel_file_path = r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx'  # Update with your file path
df = pd.read_excel(excel_file_path, sheet_name='Sheet1')

# Separate Features and Target
X = df.iloc[:, :-3]
y = df.iloc[:, -3:]

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Models for each Disease
models = {}
for disease in y_train.columns:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train[disease])
    models[disease] = rf

# Step 5: Evaluate the Models and Plot Feature Importances for Each Disease
top_n = 10

for disease, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for Disease {disease}:")
    print(classification_report(y_test[disease], y_pred))
    print(f"\nConfusion Matrix for Disease {disease}:")
    print(confusion_matrix(y_test[disease], y_pred))
    
    importances = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances for Disease {disease}")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"feature_importances_{disease.replace(' ', '_')}.png")
    plt.show()
    
    # Visualize a single tree from the Random Forest
    visualize_tree(model, feature_names, disease, tree_index=0)
    
    # Plot Partial Dependence Plots
    plt.figure(figsize=(12, 6))
    plot_partial_dependence(model, X_test, features=indices[:top_n], feature_names=feature_names, grid_resolution=50)
    plt.title(f"Partial Dependence Plots for Disease {disease}")
    plt.tight_layout()
    plt.savefig(f"pdp_{disease.replace(' ', '_')}.png")
    plt.show()

# Visualizing the whole forest
for disease, model in models.items():
    n_trees = len(model.estimators_)
    plt.figure(figsize=(15, 7))
    plt.bar(range(n_trees), [tree.tree_.node_count for tree in model.estimators_], align="center")
    plt.xlabel("Tree index")
    plt.ylabel("Number of nodes")
    plt.title(f"Number of nodes in each tree for {disease}")
    plt.savefig(f"forest_summary_{disease.replace(' ', '_')}.png")
    plt.show()


# In[4]:


#Hepatitis finale
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X, y, disease_name):
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_res, y_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_viral = train_and_evaluate(X, y_viral, 'Viral Hepatitis')
results_acute = train_and_evaluate(X, y_acute, 'Acute Hepatitis')
results_chronic = train_and_evaluate(X, y_chronic, 'Chronic Hepatitis')

# Plotting the results
def plot_results(results, disease_name, metric):
    labels = list(results.keys())
    scores = [results[model][metric] for model in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars = ax.bar(x, scores, width)

    ax.set_xlabel('Models')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.bar_label(bars, padding=3)

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Plot accuracy, precision, and recall for each disease
for disease_name, results in zip(['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'],
                                 [results_viral, results_acute, results_chronic]):
    plot_results(results, disease_name, 'accuracy')
    plot_results(results, disease_name, 'precision')
    plot_results(results, disease_name, 'recall')

# Plotting misclassification and correct classification
def plot_classification_summary(results, disease_name):
    labels = list(results.keys())
    metrics = ['Correctly Classified', 'Misclassified No Disease', 'Misclassified Disease']
    
    summary = {label: {metric: 0 for metric in metrics} for label in labels}
    
    for label in labels:
        y_test = results[label]['y_test']
        y_pred = results[label]['y_pred']
        confusion = confusion_matrix(y_test, y_pred)
        
        summary[label]['Correctly Classified'] = np.sum(np.diag(confusion))
        summary[label]['Misclassified No Disease'] = confusion[0, 1]
        summary[label]['Misclassified Disease'] = confusion[1, 0]

    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [summary[label][metric] for label in labels]
        ax.bar(x + i*width, values, width, label=metric)

    ax.set_xlabel('Models')
    ax.set_ylabel('Counts')
    ax.set_title(f'Classification Summary for {disease_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Plot classification summaries for each disease
for disease_name, results in zip(['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'],
                                 [results_viral, results_acute, results_chronic]):
    plot_classification_summary(results, disease_name)

# Plotting ROC curves
def plot_roc_curves(results, disease_name):
    fig, ax = plt.subplots()

    for name, result in results.items():
        y_test = result['y_test']
        X_test = result['X_test']
        y_pred_proba = result['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic for {disease_name}')
    ax.legend(loc='lower right')

    plt.show()

# Plot ROC curves for each disease
for disease_name, results in zip(['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'],
                                 [results_viral, results_acute, results_chronic]):
    plot_roc_curves(results, disease_name)


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X, y, disease_name):
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_res, y_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'X_test': X_test,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision: {precision:.2f}')
        print(f'Recall: {recall:.2f}')
        print(f'F1 Score: {f1:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_viral = train_and_evaluate(X, y_viral, 'Viral Hepatitis')
results_acute = train_and_evaluate(X, y_acute, 'Acute Hepatitis')
results_chronic = train_and_evaluate(X, y_chronic, 'Chronic Hepatitis')

# Plotting the results
def plot_combined_results(results, disease_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    labels = list(results.keys())
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, metric in enumerate(metrics):
        scores = [results[model][metric] for model in labels]
        ax.bar(x + i * width, scores, width, label=metric.capitalize())
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Performance Comparison for {disease_name}')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.xticks(rotation=45)
    plt.show()

# Plot combined results for each disease
for disease_name, results in zip(['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'],
                                 [results_viral, results_acute, results_chronic]):
    plot_combined_results(results, disease_name)

# The ROC curve plots remain unchanged


# In[7]:


def plot_classification_summary(results, disease_name):
    labels = list(results.keys())
    metrics = ['Correctly Classified', 'Misclassified No Disease', 'Misclassified Disease']
    
    summary = {label: {metric: 0 for metric in metrics} for label in labels}
    
    for label in labels:
        y_test = results[label]['y_test']
        y_pred = results[label]['y_pred']
        confusion = confusion_matrix(y_test, y_pred)
        
        summary[label]['Correctly Classified'] = np.sum(np.diag(confusion))
        summary[label]['Misclassified No Disease'] = confusion[0, 1]
        summary[label]['Misclassified Disease'] = confusion[1, 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [summary[label][metric] for label in labels]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xlabel('Models')
    ax.set_ylabel('Counts')
    ax.set_title(f'Classification Summary for {disease_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.xticks(rotation=45)
    plt.show()

# Plot classification summaries for each disease
for disease_name, results in zip(['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'],
                                 [results_viral, results_acute, results_chronic]):
    plot_classification_summary(results, disease_name)


# In[5]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Step 2: Display the first few rows and check data structure
print("Data Preview:")
print(df)

# Step 3: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())


# Step 5: Correlation matrix
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_viral, X_test_viral, y_train_viral, y_test_viral = train_test_split(X, y_viral, test_size=0.2, random_state=42, stratify=y_viral)
X_train_acute, X_test_acute, y_train_acute, y_test_acute = train_test_split(X, y_acute, test_size=0.2, random_state=42, stratify=y_acute)
X_train_chronic, X_test_chronic, y_train_chronic, y_test_chronic = train_test_split(X, y_chronic, test_size=0.2, random_state=42, stratify=y_chronic)

# Print train-test splits
print(f"Train-test split for Viral Hepatitis: X_train shape: {X_train_viral.shape}, X_test shape: {X_test_viral.shape}, y_train support: {len(y_train_viral)}, y_test support: {len(y_test_viral)}")
print(f"Train-test split for Acute Hepatitis: X_train shape: {X_train_acute.shape}, X_test shape: {X_test_acute.shape}, y_train support: {len(y_train_acute)}, y_test support: {len(y_test_acute)}")
print(f"Train-test split for Chronic Hepatitis: X_train shape: {X_train_chronic.shape}, X_test shape: {X_test_chronic.shape}, y_train support: {len(y_train_chronic)}, y_test support: {len(y_test_chronic)}")

# SMOTE for handling class imbalance after the split
smote = SMOTE(random_state=42)
X_train_viral_res, y_train_viral_res = smote.fit_resample(X_train_viral, y_train_viral)
X_train_acute_res, y_train_acute_res = smote.fit_resample(X_train_acute, y_train_acute)
X_train_chronic_res, y_train_chronic_res = smote.fit_resample(X_train_chronic, y_train_chronic)

# Continue with your training, evaluation, and plotting steps...


# In[18]:


# Modify plot_roc_curves to accept X_test as argument
def plot_roc_curves(results, disease_name, X_test):
    fig, ax = plt.subplots()

    for name, result in results.items():
        y_test = result['y_test']
        y_pred_proba = result['model'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'Receiver Operating Characteristic for {disease_name}')
    ax.legend(loc='lower right')

    plt.show()

# Adjust the loop to pass X_test to plot_roc_curves
for disease_name, results, X_test in zip(['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'],
                                         [results_viral, results_acute, results_chronic],
                                         [X_test_viral, X_test_acute, X_test_chronic]):
    plot_roc_curves(results, disease_name, X_test)


# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_viral, X_test_viral, y_train_viral, y_test_viral = train_test_split(X, y_viral, test_size=0.2, random_state=42, stratify=y_viral)
X_train_acute, X_test_acute, y_train_acute, y_test_acute = train_test_split(X, y_acute, test_size=0.2, random_state=42, stratify=y_acute)
X_train_chronic, X_test_chronic, y_train_chronic, y_test_chronic = train_test_split(X, y_chronic, test_size=0.2, random_state=42, stratify=y_chronic)

# Print train-test splits
print(f"Train-test split for Viral Hepatitis: X_train shape: {X_train_viral.shape}, X_test shape: {X_test_viral.shape}, y_train support: {len(y_train_viral)}, y_test support: {len(y_test_viral)}")
print(f"Train-test split for Acute Hepatitis: X_train shape: {X_train_acute.shape}, X_test shape: {X_test_acute.shape}, y_train support: {len(y_train_acute)}, y_test support: {len(y_test_acute)}")
print(f"Train-test split for Chronic Hepatitis: X_train shape: {X_train_chronic.shape}, X_test shape: {X_test_chronic.shape}, y_train support: {len(y_train_chronic)}, y_test support: {len(y_test_chronic)}")

# SMOTE for handling class imbalance after the split
smote = SMOTE(random_state=42)
X_train_viral_res, y_train_viral_res = smote.fit_resample(X_train_viral, y_train_viral)
X_train_acute_res, y_train_acute_res = smote.fit_resample(X_train_acute, y_train_acute)
X_train_chronic_res, y_train_chronic_res = smote.fit_resample(X_train_chronic, y_train_chronic)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train_res, y_train_res, X_test, y_test, disease_name):
    results = {}
    
    for name, model in models.items():
        # Calibrate the model if it supports predict_proba
        if hasattr(model, 'predict_proba'):
            # Fit the base model first
            model.fit(X_train_res, y_train_res)
            # Then calibrate using CalibratedClassifierCV
            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            calibrated_model.fit(X_train_res, y_train_res)
            model = calibrated_model
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=StratifiedKFold(n_splits=5))
        mean_cv_score = np.mean(cv_scores)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision (class 1): {precision:.2f}')
        print(f'Recall (class 1): {recall:.2f}')
        print(f'F1-score (class 1): {f1:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_viral = train_and_evaluate(X_train_viral_res, y_train_viral_res, X_test_viral, y_test_viral, 'Viral Hepatitis')
results_acute = train_and_evaluate(X_train_acute_res, y_train_acute_res, X_test_acute, y_test_acute, 'Acute Hepatitis')
results_chronic = train_and_evaluate(X_train_chronic_res, y_train_chronic_res, X_test_chronic, y_test_chronic, 'Chronic Hepatitis')

# Function to plot evaluation metrics for each disease
def plot_evaluation_metrics(results, disease_name):
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(models))
    width = 0.2  # Width of the bars
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]  # Offsets for each metric

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + offsets[i], values, width, label=metric.capitalize(), align='center')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Plot evaluation metrics for each disease
plot_evaluation_metrics(results_viral, 'Viral Hepatitis')
plot_evaluation_metrics(results_acute, 'Acute Hepatitis')
plot_evaluation_metrics(results_chronic, 'Chronic Hepatitis')

# Function to plot ROC curves for each disease
def plot_roc_curves(results, disease_name):
    plt.figure(figsize=(8, 6))
    for model, result in results.items():
        fpr = result['fpr']
        tpr = result['tpr']
        roc_auc = result['roc_auc']
        plt.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {disease_name}')
    plt.legend(loc="lower right")
    plt.show()

# Plot ROC curves for each disease
plot_roc_curves(results_viral, 'Viral Hepatitis')
plot_roc_curves(results_acute, 'Acute Hepatitis')
plot_roc_curves(results_chronic, 'Chronic Hepatitis')


# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_viral, X_test_viral, y_train_viral, y_test_viral = train_test_split(X, y_viral, test_size=0.2, random_state=42, stratify=y_viral)
X_train_acute, X_test_acute, y_train_acute, y_test_acute = train_test_split(X, y_acute, test_size=0.2, random_state=42, stratify=y_acute)
X_train_chronic, X_test_chronic, y_train_chronic, y_test_chronic = train_test_split(X, y_chronic, test_size=0.2, random_state=42, stratify=y_chronic)

# Print train-test splits
print(f"Train-test split for Viral Hepatitis: X_train shape: {X_train_viral.shape}, X_test shape: {X_test_viral.shape}, y_train support: {len(y_train_viral)}, y_test support: {len(y_test_viral)}")
print(f"Train-test split for Acute Hepatitis: X_train shape: {X_train_acute.shape}, X_test shape: {X_test_acute.shape}, y_train support: {len(y_train_acute)}, y_test support: {len(y_test_acute)}")
print(f"Train-test split for Chronic Hepatitis: X_train shape: {X_train_chronic.shape}, X_test shape: {X_test_chronic.shape}, y_train support: {len(y_train_chronic)}, y_test support: {len(y_test_chronic)}")

# SMOTE for handling class imbalance after the split
smote = SMOTE(random_state=42)
X_train_viral_res, y_train_viral_res = smote.fit_resample(X_train_viral, y_train_viral)
X_train_acute_res, y_train_acute_res = smote.fit_resample(X_train_acute, y_train_acute)
X_train_chronic_res, y_train_chronic_res = smote.fit_resample(X_train_chronic, y_train_chronic)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train_res, y_train_res, X_test, y_test, disease_name):
    results = {}
    
    for name, model in models.items():
        # Calibrate the model if it supports predict_proba
        if hasattr(model, 'predict_proba'):
            # Fit the base model first
            model.fit(X_train_res, y_train_res)
            # Then calibrate using CalibratedClassifierCV
            calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            calibrated_model.fit(X_train_res, y_train_res)
            model = calibrated_model
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=StratifiedKFold(n_splits=5))
        mean_cv_score = np.mean(cv_scores)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision (class 1): {precision:.2f}')
        print(f'Recall (class 1): {recall:.2f}')
        print(f'F1-score (class 1): {f1:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_viral = train_and_evaluate(X_train_viral_res, y_train_viral_res, X_test_viral, y_test_viral, 'Viral Hepatitis')
results_acute = train_and_evaluate(X_train_acute_res, y_train_acute_res, X_test_acute, y_test_acute, 'Acute Hepatitis')
results_chronic = train_and_evaluate(X_train_chronic_res, y_train_chronic_res, X_test_chronic, y_test_chronic, 'Chronic Hepatitis')

# Function to plot evaluation metrics for each disease
def plot_evaluation_metrics(results, disease_name):
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(models))
    width = 0.2  # Width of the bars
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]  # Offsets for each metric

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + offsets[i], values, width, label=metric.capitalize(), align='center')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Plot evaluation metrics for each disease
plot_evaluation_metrics(results_viral, 'Viral Hepatitis')
plot_evaluation_metrics(results_acute, 'Acute Hepatitis')
plot_evaluation_metrics(results_chronic, 'Chronic Hepatitis')

# Function to plot ROC curve for all models for a disease
def plot_roc_curves(results, disease_name):
    plt.figure(figsize=(10, 6))

    for name, result in results.items():
        # Plot the ROC curve only if there are at least two distinct values in y_proba
        if len(np.unique(result['y_proba'])) > 1:
            plt.plot(result['fpr'], result['tpr'], label=f'{name} (AUC = {result["roc_auc"]:.2f})')
        else:
            print(f"Skipping {name} for {disease_name} due to lack of distinct predicted probabilities.")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {disease_name}')
    plt.legend(loc='lower right')
    plt.show()

# Plot ROC curves for each disease
plot_roc_curves(results_viral, 'Viral Hepatitis')
plot_roc_curves(results_acute, 'Acute Hepatitis')
plot_roc_curves(results_chronic, 'Chronic Hepatitis')


# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_viral, X_test_viral, y_train_viral, y_test_viral = train_test_split(X, y_viral, test_size=0.2, random_state=42, stratify=y_viral)
X_train_acute, X_test_acute, y_train_acute, y_test_acute = train_test_split(X, y_acute, test_size=0.2, random_state=42, stratify=y_acute)
X_train_chronic, X_test_chronic, y_train_chronic, y_test_chronic = train_test_split(X, y_chronic, test_size=0.2, random_state=42, stratify=y_chronic)

# Print train-test splits
print(f"Train-test split for Viral Hepatitis: X_train shape: {X_train_viral.shape}, X_test shape: {X_test_viral.shape}, y_train support: {len(y_train_viral)}, y_test support: {len(y_test_viral)}")
print(f"Train-test split for Acute Hepatitis: X_train shape: {X_train_acute.shape}, X_test shape: {X_test_acute.shape}, y_train support: {len(y_train_acute)}, y_test support: {len(y_test_acute)}")
print(f"Train-test split for Chronic Hepatitis: X_train shape: {X_train_chronic.shape}, X_test shape: {X_test_chronic.shape}, y_train support: {len(y_train_chronic)}, y_test support: {len(y_test_chronic)}")

# SMOTE for handling class imbalance after the split
smote = SMOTE(random_state=42)
X_train_viral_res, y_train_viral_res = smote.fit_resample(X_train_viral, y_train_viral)
X_train_acute_res, y_train_acute_res = smote.fit_resample(X_train_acute, y_train_acute)
X_train_chronic_res, y_train_chronic_res = smote.fit_resample(X_train_chronic, y_train_chronic)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train_res, y_train_res, X_test, y_test, disease_name):
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision (class 1): {precision:.2f}')
        print(f'Recall (class 1): {recall:.2f}')
        print(f'F1-score (class 1): {f1:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_viral = train_and_evaluate(X_train_viral_res, y_train_viral_res, X_test_viral, y_test_viral, 'Viral Hepatitis')
results_acute = train_and_evaluate(X_train_acute_res, y_train_acute_res, X_test_acute, y_test_acute, 'Acute Hepatitis')
results_chronic = train_and_evaluate(X_train_chronic_res, y_train_chronic_res, X_test_chronic, y_test_chronic, 'Chronic Hepatitis')

# Function to plot evaluation metrics for each disease
def plot_evaluation_metrics(results, disease_name):
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(models))
    width = 0.2  # Width of the bars
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]  # Offsets for each metric

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + offsets[i], values, width, label=metric.capitalize(), align='center')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Plot evaluation metrics for each disease
plot_evaluation_metrics(results_viral, 'Viral Hepatitis')
plot_evaluation_metrics(results_acute, 'Acute Hepatitis')
plot_evaluation_metrics(results_chronic, 'Chronic Hepatitis')


# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Load and preprocess data
data = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Assuming the last 3 columns are the target variables (diseases)
X = data.drop(columns=['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'])
y = data[['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis']]

# Step 2: Initialize lists to store resampled data
X_resampled_list = []
y_resampled_list = []

# Step 3: Apply SMOTE to each disease column separately
smote = SMOTE(random_state=42)

for i in range(y.shape[1]):
    X_resampled_i, y_resampled_i = smote.fit_resample(X, y.iloc[:, i])
    X_resampled_list.append(X_resampled_i)
    y_resampled_list.append(y_resampled_i)

# Step 4: Split resampled data into training and testing sets
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

for i in range(y.shape[1]):
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_resampled_list[i], y_resampled_list[i], test_size=0.2, random_state=42)
    X_train_list.append(X_train_i)
    X_test_list.append(X_test_i)
    y_train_list.append(y_train_i)
    y_test_list.append(y_test_i)

# Step 5: Initialize Decision Tree Classifier and train for each disease
dt_models = []

for i in range(y.shape[1]):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_list[i], y_train_list[i])
    dt_models.append(dt)
    
    # Step 6: Evaluate the model for each disease
    y_pred_i = dt.predict(X_test_list[i])
    
    print(f"Classification Report for {y.columns[i]}:")
    print(classification_report(y_test_list[i], y_pred_i))
    
    cm = confusion_matrix(y_test_list[i], y_pred_i)
    print(f"\nConfusion Matrix for {y.columns[i]}:")
    print(cm)
    print("\n")
    
    # Plot the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"Confusion Matrix for {y.columns[i]}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Step 7: Visualize Decision Tree
    plt.figure(figsize=(15, 10))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['0', '1'])
    plt.title(f"Decision Tree for {y.columns[i]} Prediction")
    plt.show()

# Step 8: Visualize feature importances for the last trained model (Chronic Hepatitis)
plt.figure(figsize=(10, 6))
importances = pd.Series(dt_models[-1].feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.show()


# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Step 1: Load and preprocess data
data = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Assuming the last 3 columns are the target variables (diseases)
X = data.drop(columns=['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'])
y = data[['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis']]

# Step 2: Initialize lists to store resampled data
X_resampled_list = []
y_resampled_list = []

# Step 3: Apply SMOTE to each disease column separately
smote = SMOTE(random_state=42)

for i in range(y.shape[1]):
    X_resampled_i, y_resampled_i = smote.fit_resample(X, y.iloc[:, i])
    X_resampled_list.append(X_resampled_i)
    y_resampled_list.append(y_resampled_i)

# Step 4: Split resampled data into training and testing sets
X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

for i in range(y.shape[1]):
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_resampled_list[i], y_resampled_list[i], train_size=68, test_size=17, random_state=42
    )
    X_train_list.append(X_train_i)
    X_test_list.append(X_test_i)
    y_train_list.append(y_train_i)
    y_test_list.append(y_test_i)

# Step 5: Initialize Decision Tree Classifier and train for each disease
dt_models = []

for i in range(y.shape[1]):
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_list[i], y_train_list[i])
    dt_models.append(dt)
    
    # Step 6: Evaluate the model for each disease
    y_pred_i = dt.predict(X_test_list[i])
    
    print(f"Classification Report for {y.columns[i]}:")
    print(classification_report(y_test_list[i], y_pred_i))
    
    cm = confusion_matrix(y_test_list[i], y_pred_i)
    print(f"\nConfusion Matrix for {y.columns[i]}:")
    print(cm)
    print("\n")
    
    # Plot the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"Confusion Matrix for {y.columns[i]}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Step 7: Visualize Decision Tree
    plt.figure(figsize=(15, 10))
    plot_tree(dt, filled=True, feature_names=X.columns, class_names=['0', '1'])
    plt.title(f"Decision Tree for {y.columns[i]} Prediction")
    plt.show()

# Step 8: Visualize feature importances for the last trained model (Chronic Hepatitis)
plt.figure(figsize=(10, 6))
importances = pd.Series(dt_models[-1].feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.xlabel("Feature Importance")
plt.show()


# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import graphviz
from sklearn.tree import export_graphviz
from sklearn.inspection import plot_partial_dependence

# Define the visualize_tree function
def visualize_tree(model, feature_names, target, tree_index=0):
    estimator = model.estimators_[tree_index]
    dot_data = export_graphviz(estimator, out_file=None,
                               feature_names=feature_names,
                               class_names=[str(class_name) for class_name in model.classes_],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    output_file_pdf = f"random_forest_tree_{target.replace(' ', '_')}.pdf"
    graph.render(output_file_pdf, format='pdf')
    output_file_png = f"random_forest_tree_{target.replace(' ', '_')}.png"
    graph.render(output_file_png, format='png')
    graph.view()

# Load the data from Excel
excel_file_path = r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx'  # Update with your file path
df = pd.read_excel(excel_file_path, sheet_name='Sheet1')

# Separate Features and Target
X = df.iloc[:, :-3]
y = df.iloc[:, -3:]

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=68, test_size=17, random_state=42)

# Step 4: Train Random Forest Models for each Disease
models = {}
for disease in y_train.columns:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train[disease])
    models[disease] = rf

# Step 5: Evaluate the Models and Plot Feature Importances for Each Disease
top_n = 10

for disease, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\nClassification Report for Disease {disease}:")
    print(classification_report(y_test[disease], y_pred))
    
    cm = confusion_matrix(y_test[disease], y_pred)
    print(f"\nConfusion Matrix for Disease {disease}:")
    print(cm)
    
    # Plot the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"Confusion Matrix for {disease}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    importances = model.feature_importances_
    feature_names = X.columns
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top {top_n} Feature Importances for Disease {disease}")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center")
    plt.xticks(range(top_n), feature_names[indices[:top_n]], rotation=90)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(f"feature_importances_{disease.replace(' ', '_')}.png")
    plt.show()
    
    # Visualize a single tree from the Random Forest
    visualize_tree(model, feature_names, disease, tree_index=0)
    
    # Plot Partial Dependence Plots
    plt.figure(figsize=(12, 6))
    plot_partial_dependence(model, X_test, features=indices[:top_n], feature_names=feature_names, grid_resolution=50)
    plt.title(f"Partial Dependence Plots for Disease {disease}")
    plt.tight_layout()
    plt.savefig(f"pdp_{disease.replace(' ', '_')}.png")
    plt.show()

# Visualizing the whole forest
for disease, model in models.items():
    n_trees = len(model.estimators_)
    plt.figure(figsize=(15, 7))
    plt.bar(range(n_trees), [tree.tree_.node_count for tree in model.estimators_], align="center")
    plt.xlabel("Tree index")
    plt.ylabel("Number of nodes")
    plt.title(f"Number of nodes in each tree for {disease}")
    plt.savefig(f"forest_summary_{disease.replace(' ', '_')}.png")
    plt.show()


# In[30]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.inspection import plot_partial_dependence

# Load the data from Excel
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx', sheet_name='Sheet1')  # Adjust sheet_name as necessary

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last three)
y = df.iloc[:, -3:]  # Target (last three columns)

# Train and evaluate models for each target
for target in y.columns:
    # Split the data into training (68 samples) and testing (17 samples) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y[target], train_size=68, test_size=17, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # 1. Plot feature importance
    coefficients = model.coef_[0]
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coefficients)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance in Logistic Regression for {target}')
    plt.show()

    # 2. Predict probabilities and plot ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Logistic Regression - {target}')
    plt.legend()
    plt.show()

    # 3. Plot partial dependence for a specific feature (example for the first feature)
    plt.figure(figsize=(10, 6))
    plot_partial_dependence(model, X_train, features=[0], feature_names=feature_names)
    plt.xlabel('Feature Value')
    plt.ylabel('Partial Dependence')
    plt.title(f'Partial Dependence Plot for Feature 1 - {target}')
    plt.show()

    # 4. Plot the Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"Confusion Matrix for {target}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 5. Print classification report
    print(f"\nClassification Report for Disease {target}:")
    print(classification_report(y_test, y_pred))


# In[33]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.inspection import plot_partial_dependence

# Load the data from Excel
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx', sheet_name='Sheet1')  # Adjust sheet_name as necessary

# Separate Features and Target
X = df.iloc[:, :-3]  # Features (all columns except the last three)
y = df.iloc[:, -3:]  # Target (last three columns)

# Train and evaluate models for each target
for target in y.columns:
    # Split the data into training (68 samples) and testing (17 samples) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y[target], train_size=68, test_size=17, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # 1. Plot feature importance
    coefficients = model.coef_[0]
    feature_names = X.columns
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, coefficients)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance in Logistic Regression for {target}')
    plt.show()

    # 2. Predict probabilities and plot ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_pred_proba):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Logistic Regression - {target}')
    plt.legend()
    plt.show()

    # 3. Plot partial dependence for a specific feature (example for the first feature)
    plt.figure(figsize=(10, 6))
    plot_partial_dependence(model, X_train, features=[0], feature_names=feature_names)
    plt.xlabel('Feature Value')
    plt.ylabel('Partial Dependence')
    plt.title(f'Partial Dependence Plot for Feature 1 - {target}')
    plt.show()

    # 4. Plot the Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f"Confusion Matrix for {target}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 5. Print classification report
    print(f"\nClassification Report for Disease {target} (focus on class 1):")
    report = classification_report(y_test, y_pred, target_names=['class 0', 'class 1'], output_dict=True)
    print(f"Precision for class 1: {report['class 1']['precision']:.2f}")
    print(f"Recall for class 1: {report['class 1']['recall']:.2f}")
    print(f"F1-score for class 1: {report['class 1']['f1-score']:.2f}")
    print(f"Support for class 1: {report['class 1']['support']}\n")

    # Accuracy
    accuracy = report['accuracy']
    print(f"Accuracy: {accuracy:.2f}\n")

    # Specific metrics for class 1
    precision_1 = report['class 1']['precision']
    recall_1 = report['class 1']['recall']
    f1_score_1 = report['class 1']['f1-score']

    # Output metrics
    print(f"Class 1 Performance Metrics for {target}:")
    print(f"Precision: {precision_1:.2f}")
    print(f"Recall: {recall_1:.2f}")
    print(f"F1-score: {f1_score_1:.2f}")
    print(f"Support: {report['class 1']['support']}\n")


# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_viral, X_test_viral, y_train_viral, y_test_viral = train_test_split(X, y_viral, test_size=0.2, random_state=42, stratify=y_viral)
X_train_acute, X_test_acute, y_train_acute, y_test_acute = train_test_split(X, y_acute, test_size=0.2, random_state=42, stratify=y_acute)
X_train_chronic, X_test_chronic, y_train_chronic, y_test_chronic = train_test_split(X, y_chronic, test_size=0.2, random_state=42, stratify=y_chronic)

# Print train-test splits
print(f"Train-test split for Viral Hepatitis: X_train shape: {X_train_viral.shape}, X_test shape: {X_test_viral.shape}, y_train support: {len(y_train_viral)}, y_test support: {len(y_test_viral)}")
print(f"Train-test split for Acute Hepatitis: X_train shape: {X_train_acute.shape}, X_test shape: {X_test_acute.shape}, y_train support: {len(y_train_acute)}, y_test support: {len(y_test_acute)}")
print(f"Train-test split for Chronic Hepatitis: X_train shape: {X_train_chronic.shape}, X_test shape: {X_test_chronic.shape}, y_train support: {len(y_train_chronic)}, y_test support: {len(y_test_chronic)}")

# SMOTE for handling class imbalance after the split
smote = SMOTE(random_state=42)
X_train_viral_res, y_train_viral_res = smote.fit_resample(X_train_viral, y_train_viral)
X_train_acute_res, y_train_acute_res = smote.fit_resample(X_train_acute, y_train_acute)
X_train_chronic_res, y_train_chronic_res = smote.fit_resample(X_train_chronic, y_train_chronic)

# Define a dictionary to hold the models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
}

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train_res, y_train_res, X_test, y_test, disease_name):
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
        recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
        f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
        cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
        mean_cv_score = np.mean(cv_scores)
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_cv_score': mean_cv_score,
            'y_test': y_test,
            'y_pred': y_pred,
            'model': model
        }
        print(f'Evaluation for {disease_name} using {name}:')
        print(f'Accuracy: {accuracy:.2f}')
        print(f'Precision (class 1): {precision:.2f}')
        print(f'Recall (class 1): {recall:.2f}')
        print(f'F1-score (class 1): {f1:.2f}')
        print(f'Mean CV Score: {mean_cv_score:.2f}')
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('')

    return results

# Train and evaluate models for each disease
results_viral = train_and_evaluate(X_train_viral_res, y_train_viral_res, X_test_viral, y_test_viral, 'Viral Hepatitis')
results_acute = train_and_evaluate(X_train_acute_res, y_train_acute_res, X_test_acute, y_test_acute, 'Acute Hepatitis')
results_chronic = train_and_evaluate(X_train_chronic_res, y_train_chronic_res, X_test_chronic, y_test_chronic, 'Chronic Hepatitis')

# Function to plot evaluation metrics for each disease
def plot_evaluation_metrics(results, disease_name):
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(models))
    width = 0.2  # Width of the bars
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]  # Offsets for each metric

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in models]
        ax.bar(x + offsets[i], values, width, label=metric.capitalize(), align='center')

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Plot evaluation metrics for each disease
plot_evaluation_metrics(results_viral, 'Viral Hepatitis')
plot_evaluation_metrics(results_acute, 'Acute Hepatitis')
plot_evaluation_metrics(results_chronic, 'Chronic Hepatitis')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {title}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrices for each disease and model
for disease_name, results in zip(['Viral Hepatitis', 'Acute Hepatitis', 'Chronic Hepatitis'], [results_viral, results_acute, results_chronic]):
    for model_name in results.keys():
        cm = confusion_matrix(results[model_name]['y_test'], results[model_name]['y_pred'])
        plot_confusion_matrix(cm, ['No', 'Yes'], f"{disease_name} using {model_name}")


# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_excel(r'C:\Users\gangu\OneDrive\Documents\Male below 45 more.xlsx')

# Separate features and target variables
X = df.iloc[:, :-3]  # All columns except the last three
y_viral = df.iloc[:, -3]  # Third last column
y_acute = df.iloc[:, -2]  # Second last column
y_chronic = df.iloc[:, -1]  # Last column

# Print original dataset size
print(f"Original dataset size: {len(df)} samples")

# Use train_test_split with stratify
X_train_viral, X_test_viral, y_train_viral, y_test_viral = train_test_split(X, y_viral, test_size=0.2, random_state=42, stratify=y_viral)
X_train_acute, X_test_acute, y_train_acute, y_test_acute = train_test_split(X, y_acute, test_size=0.2, random_state=42, stratify=y_acute)
X_train_chronic, X_test_chronic, y_train_chronic, y_test_chronic = train_test_split(X, y_chronic, test_size=0.2, random_state=42, stratify=y_chronic)

# Print train-test splits
print(f"Train-test split for Viral Hepatitis: X_train shape: {X_train_viral.shape}, X_test shape: {X_test_viral.shape}, y_train support: {len(y_train_viral)}, y_test support: {len(y_test_viral)}")
print(f"Train-test split for Acute Hepatitis: X_train shape: {X_train_acute.shape}, X_test shape: {X_test_acute.shape}, y_train support: {len(y_train_acute)}, y_test support: {len(y_test_acute)}")
print(f"Train-test split for Chronic Hepatitis: X_train shape: {X_train_chronic.shape}, X_test shape: {X_test_chronic.shape}, y_train support: {len(y_train_chronic)}, y_test support: {len(y_test_chronic)}")

# SMOTE for handling class imbalance after the split
smote = SMOTE(random_state=42)
X_train_viral_res, y_train_viral_res = smote.fit_resample(X_train_viral, y_train_viral)
X_train_acute_res, y_train_acute_res = smote.fit_resample(X_train_acute, y_train_acute)
X_train_chronic_res, y_train_chronic_res = smote.fit_resample(X_train_chronic, y_train_chronic)

# Gaussian Naive Bayes model
model = GaussianNB()

# Helper function to train, evaluate and return the results
def train_and_evaluate(X_train_res, y_train_res, X_test, y_test, disease_name):
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1, average='binary')
    recall = recall_score(y_test, y_pred, pos_label=1, average='binary')
    f1 = f1_score(y_test, y_pred, pos_label=1, average='binary')
    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=5)
    mean_cv_score = np.mean(cv_scores)
    
    print(f'Evaluation for {disease_name} using Gaussian Naive Bayes:')
    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision (class 1): {precision:.2f}')
    print(f'Recall (class 1): {recall:.2f}')
    print(f'F1-score (class 1): {f1:.2f}')
    print(f'Mean CV Score: {mean_cv_score:.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_cv_score': mean_cv_score,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }

# Train and evaluate models for each disease
results_viral = train_and_evaluate(X_train_viral_res, y_train_viral_res, X_test_viral, y_test_viral, 'Viral Hepatitis')
results_acute = train_and_evaluate(X_train_acute_res, y_train_acute_res, X_test_acute, y_test_acute, 'Acute Hepatitis')
results_chronic = train_and_evaluate(X_train_chronic_res, y_train_chronic_res, X_test_chronic, y_test_chronic, 'Chronic Hepatitis')

# Function to plot evaluation metrics for each disease
def plot_evaluation_metrics(results, disease_name):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    scores = {metric: [results[metric]] for metric in metrics}

    labels = ['Gaussian Naive Bayes']
    x = np.arange(len(labels))
    width = 0.2  # Width of the bars
    offset = -0.3

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        bars = ax.bar(x + offset + i*width, scores[metric], width, label=metric.capitalize())

    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title(f'Evaluation Metrics for {disease_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.xticks(rotation=45)
    plt.show()

# Plot evaluation metrics for each disease
plot_evaluation_metrics(results_viral, 'Viral Hepatitis')
plot_evaluation_metrics(results_acute, 'Acute Hepatitis')
plot_evaluation_metrics(results_chronic, 'Chronic Hepatitis')

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix for {title}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Plot confusion matrices for each disease
plot_confusion_matrix(confusion_matrix(results_viral['y_test'], results_viral['y_pred']), ['No', 'Yes'], 'Viral Hepatitis')
plot_confusion_matrix(confusion_matrix(results_acute['y_test'], results_acute['y_pred']), ['No', 'Yes'], 'Acute Hepatitis')
plot_confusion_matrix(confusion_matrix(results_chronic['y_test'], results_chronic['y_pred']), ['No', 'Yes'], 'Chronic Hepatitis')

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_prob, disease_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for {disease_name}')
    plt.legend(loc='lower right')
    plt.show()

# Plot ROC curves for each disease
plot_roc_curve(results_viral['y_test'], results_viral['y_pred_prob'], 'Viral Hepatitis')
plot_roc_curve(results_acute['y_test'], results_acute['y_pred_prob'], 'Acute Hepatitis')
plot_roc_curve(results_chronic['y_test'], results_chronic['y_pred_prob'], 'Chronic Hepatitis')


# In[ ]:




