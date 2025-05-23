# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np # For handling potential NaN if any step introduces them

# --- Configuration ---
# These column names MUST EXACTLY MATCH your student_answers.csv headers
QUESTION_COLUMNS = [
    "subject_group",
    "subject_enjoyment",
    "learning_style",
    "hobby",
    "interest_field",
    "problem_approach",
    "work_environment",
    "decision_making",
    "team_role",
    "motivation",
    "longterm_goal"
]
TARGET_COLUMN = "chosen_major" # This is what the AI will learn to predict
DATA_PATH = 'student_answers.csv'

MODEL_SAVE_PATH = 'major_recommender_model.pkl'
ENCODERS_SAVE_PATH = 'label_encoders.pkl'

# --- Preprocessing Function ---
def preprocess_data(df, feature_columns, target_column):
    """
    Preprocesses the data:
    - Handles missing values (simple fill with 'Unknown').
    - Encodes categorical features and the target variable using LabelEncoder.
    - Returns processed features (X), processed target (y), and encoders.
    """
    logger.info("Starting data preprocessing...")

    # Create a copy to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # Handle missing values for feature columns (simple strategy: fill with a placeholder)
    for col in feature_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna('Unknown').astype(str)
        else:
            logger.error(f"Feature column '{col}' not found in DataFrame.")
            # Optionally raise an error or handle as appropriate
            raise ValueError(f"Feature column '{col}' not found.")

    # Handle missing values for target column
    if target_column in df_processed.columns:
        df_processed[target_column] = df_processed[target_column].fillna('Unknown').astype(str)
    else:
        logger.error(f"Target column '{target_column}' not found in DataFrame.")
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df_processed[feature_columns]
    y = df_processed[target_column]

    label_encoders = {}

    # Encode feature columns
    logger.info("Encoding feature columns...")
    for column in X.columns:
        le = LabelEncoder()
        X.loc[:, column] = le.fit_transform(X[column])  # Fix SettingWithCopyWarning
        label_encoders[column] = le
        logger.debug(f"Encoded column: {column}")

    # Encode target column
    logger.info("Encoding target column...")
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    label_encoders[target_column] = le_target # Save with the original target column name
    logger.debug(f"Encoded target column: {target_column}")

    logger.info("Data preprocessing complete.")
    return X, y_encoded, label_encoders

# --- Model Training Function ---
def train_and_evaluate_model(X, y):
    """
    Splits data, trains a RandomForestClassifier, and evaluates it.
    Returns the trained model.
    """
    logger.info("Splitting data into training and testing sets...")
    # Ensure test_size is at least the number of classes
    n_classes = len(np.unique(y))
    min_test_size = max(n_classes, int(0.2 * len(y)))
    if min_test_size >= len(y):
        raise ValueError(f"Not enough samples to split: {len(y)} samples, {n_classes} classes.")
    test_size = min_test_size / len(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if n_classes > 1 else None
    )
    logger.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    logger.info("Training RandomForestClassifier model...")
    # You can tune these parameters: n_estimators, max_depth, etc.
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    logger.info("Model training complete.")

    # Evaluate the model
    y_pred_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    logger.info(f"Model Test Accuracy: {accuracy:.4f}")

    # Classification report (provides precision, recall, f1-score per class)
    # Ensure labels for classification_report are from the original string values if possible, or encoded if not
    try:
        # Get target encoder using the TARGET_COLUMN key
        le_target = label_encoders.get(TARGET_COLUMN) # label_encoders needs to be accessible here
        if le_target:
             report_labels_encoded = np.unique(np.concatenate((y_test, y_pred_test)))
             report_labels_string = le_target.inverse_transform(report_labels_encoded)
             report = classification_report(y_test, y_pred_test, labels=report_labels_encoded, target_names=report_labels_string, zero_division=0)
             logger.info("Classification Report:\n" + report)
        else:
            logger.warning("Target label encoder not found for classification report string labels.")
            report = classification_report(y_test, y_pred_test, zero_division=0)
            logger.info("Classification Report (encoded labels):\n" + report)

    except Exception as e:
        logger.error(f"Could not generate detailed classification report: {e}")
        # Fallback to simple accuracy if report fails
        # (Accuracy already logged above)

    return model

# --- Main Execution ---
if __name__ == "__main__":
    # Setup basic logging for this script
    import logging as script_logging # Use a different alias if main_bot's logger is different
    script_logging.basicConfig(level=script_logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = script_logging.getLogger(__name__)

    logger.info(f"Loading data from: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info("Data loaded successfully.")
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        logger.info(f"Dataset shape: {df.shape}")

        # Verify all QUESTION_COLUMNS and TARGET_COLUMN exist in the DataFrame
        missing_q_cols = [col for col in QUESTION_COLUMNS if col not in df.columns]
        if missing_q_cols:
            logger.error(f"Missing feature columns in CSV: {missing_q_cols}")
            raise ValueError(f"Missing feature columns in CSV: {missing_q_cols}")
        if TARGET_COLUMN not in df.columns:
            logger.error(f"Target column '{TARGET_COLUMN}' not found in CSV.")
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV.")

        # Proceed with preprocessing
        X_processed, y_processed, label_encoders = preprocess_data(df, QUESTION_COLUMNS, TARGET_COLUMN)

        # Train the model
        trained_model = train_and_evaluate_model(X_processed, y_processed)

        # Save the model and encoders
        logger.info(f"Saving model to: {MODEL_SAVE_PATH}")
        joblib.dump(trained_model, MODEL_SAVE_PATH)
        logger.info(f"Saving label encoders to: {ENCODERS_SAVE_PATH}")
        joblib.dump(label_encoders, ENCODERS_SAVE_PATH)

        logger.info("--- Training process completed successfully! ---")

    except FileNotFoundError:
        logger.error(f"Error: The data file '{DATA_PATH}' was not found.")
    except ValueError as ve:
        logger.error(f"ValueError during script execution: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)