import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the base directory path
base_dir = '/Users/md.shamsouddinbhuiyanshuvo/Desktop/Data_Preparation_Project'
data_preparation_dir = os.path.join(base_dir, 'Data_Preparation')
training_testing_sets_dir = os.path.join(base_dir, 'Training_Testing_Sets')
scaling_documentation_dir = os.path.join(base_dir, 'Scaling_Documentation')

# Ensure the directories exist
os.makedirs(data_preparation_dir, exist_ok=True)
os.makedirs(training_testing_sets_dir, exist_ok=True)
os.makedirs(scaling_documentation_dir, exist_ok=True)

# Load the dataset
file_path = os.path.join(base_dir, '/Users/md.shamsouddinbhuiyanshuvo/Desktop/Data_Preparation_Project/Data_Preparation/Dataset (ATS)-1.csv')
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Print all columns in the dataset
print("\nColumns in the dataset:")
print(data.columns)

# Check and display missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Visualize missing data
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Handle missing data: Fill missing values in numerical columns with mean
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

# Fill missing values in categorical columns with the mode (most frequent value)
categorical_columns_with_na = data.select_dtypes(include=['object']).columns
for col in categorical_columns_with_na:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Display the dataset after handling missing values
print("\nData after handling missing values:")
print(data.head())

# Encode categorical variables
categorical_columns = [
    'gender', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
    'Contract', 'Churn'
]

# Ensure only existing categorical columns are encoded
existing_categorical_columns = [col for col in categorical_columns if col in data.columns]
print("\nCategorical columns to be encoded:", existing_categorical_columns)

data_encoded = pd.get_dummies(data, columns=existing_categorical_columns, drop_first=True)

# Remove duplicates
data_encoded = data_encoded.drop_duplicates()

# Ensure proper data types
if 'tenure' in data_encoded.columns:
    data_encoded['tenure'] = data_encoded['tenure'].astype(int)

if 'MonthlyCharges' in data_encoded.columns:
    data_encoded['MonthlyCharges'] = data_encoded['MonthlyCharges'].astype(float)

# Save the preprocessed dataset
try:
    data_encoded.to_csv(os.path.join(data_preparation_dir, 'preprocessed_dataset.csv'), index=False)
    print("\nPreprocessed dataset saved as 'preprocessed_dataset.csv'.")
except OSError as e:
    print(f"Error saving preprocessed dataset: {e}")

# Load the Preprocessed Dataset for Stage 2
data_encoded = pd.read_csv(os.path.join(data_preparation_dir, 'preprocessed_dataset.csv'))

# Ensure all necessary categorical columns are present
required_features = [
    'tenure', 'MonthlyCharges',
    'InternetService_Fiber optic', 'PhoneService_Yes', 'MultipleLines_Yes',
    'Contract_Two year', 'gender_Male', 'Dependents_Yes', 'Churn_Yes'
]

# Check if all required features are present after encoding
missing_features = [feature for feature in required_features if feature not in data_encoded.columns]
if missing_features:
    raise ValueError(f"Missing features in the dataset: {missing_features}")

# Select the relevant features
data_selected = data_encoded[required_features]

# Display the selected features
print("\nSelected features for analysis:")
print(data_selected.head())

# Engineer New Features or Transform Existing Ones
data_encoded['TenureGroup'] = pd.cut(
    data_encoded['tenure'],
    bins=[0, 12, 24, 48, 72, 1000],
    labels=['0-1yr', '1-2yrs', '2-4yrs', '4-6yrs', '6+yrs']
)

data_encoded['MonthlyChargesBin'] = pd.cut(
    data_encoded['MonthlyCharges'],
    bins=[0, 30, 60, 90, 120],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# Display the new features
print("\nEngineered features:")
print(data_encoded[['TenureGroup', 'MonthlyChargesBin']].head())

# Handle Feature Scaling and Normalization
features_to_scale = ['tenure', 'MonthlyCharges']
scaler = StandardScaler()
data_encoded[features_to_scale] = scaler.fit_transform(data_encoded[features_to_scale])

# Display the scaled features
print("\nScaled features:")
print(data_encoded[features_to_scale].head())

# Split the dataset into training and testing sets
X = data_encoded.drop(columns=['Churn_Yes'])
y = data_encoded['Churn_Yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the training and testing sets
try:
    X_train.to_csv(os.path.join(training_testing_sets_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(training_testing_sets_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(training_testing_sets_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(training_testing_sets_dir, 'y_test.csv'), index=False)
    print("\nTraining and testing sets saved.")
except OSError as e:
    print(f"Error saving training and testing sets: {e}")

# Save the preprocessed dataset for Stage 2
try:
    data_encoded.to_csv(os.path.join(data_preparation_dir, 'preprocessed_dataset_stage2.csv'), index=False)
    print("\nPreprocessed dataset for Stage 2 saved as 'preprocessed_dataset_stage2.csv'.")
except OSError as e:
    print(f"Error saving preprocessed dataset for Stage 2: {e}")

# Document the feature engineering steps
try:
    with open(os.path.join(scaling_documentation_dir, 'feature_engineering_log.txt'), 'w') as f:
        f.write("Feature Engineering Steps:\n")
        f.write("1. Selected relevant features for analysis.\n")
        f.write("2. Created new features: Tenure groups and MonthlyCharges bins.\n")
        f.write("3. Standardized numerical features: tenure, MonthlyCharges.\n")
        f.write("4. Split the dataset into training and testing sets.\n")
        f.write("5. Saved the preprocessed data for model training.\n")
    print("Feature engineering steps documented in 'Scaling_Documentation/feature_engineering_log.txt'.")
except OSError as e:
    print(f"Error documenting feature engineering steps: {e}")
