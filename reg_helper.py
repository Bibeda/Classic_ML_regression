from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_validate


def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def train_evaluate_models_cv(models: list, X, y, preprocessor=None, cv=5, seed=None):
    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели, переданные в качестве аргументов
        current_model = clone(model)
        current_preprocessor = clone(preprocessor)

        # Store metrics
        all_metrics[model_name] = train_evaluate_model_cv(
            current_model, model_name, X, y, current_preprocessor, cv, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".2f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df


def train_evaluate_models(models: list, X_train, y_train, X_test, y_test, seed=None):
    """
    Train and evaluate multiple classification models, then display a heatmap of the metrics.

    Parameters:
    -----------
    models : list
        List of tuples containing (model_name, model_instance) where model_instance is a scikit-learn compatible classifier
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    preprocessor : Pipeline or Transformer, optional
        Preprocessing pipeline to apply to the data before training
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all evaluation metrics for all models
    """

    # Dictionary to store all metrics
    all_metrics = {}

    for model_name, model in models:
        # Работаем с копией модели, чтобы не изменять исходные модели, переданные в качестве аргументов
        current_model = clone(model)

        # Store metrics
        all_metrics[model_name] = train_evaluate_model(
            current_model, model_name, X_train, y_train, X_test, y_test, seed)

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".2f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df



def train_evaluate_model_cv(model, model_name, X, y,
                            preprocessor=None, cv=5, seed=None):
    """
    Train and evaluate a model using cross-validation and optional preprocessing.

    Args:
        model: The model to train and evaluate
        model_name: Name of the model for reporting
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        preprocessor: Preprocessing pipeline (e.g., StandardScaler, OneHotEncoder)
        cv: Number of cross-validation folds
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing evaluation metrics
    """
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Create or extend pipeline with preprocessor and model
    if isinstance(preprocessor, Pipeline):
        # If preprocessor is already a pipeline, append the model to it
        preprocessor.steps.append(('model', model))
        pipeline = preprocessor
    elif preprocessor is not None:
        # Create new pipeline with preprocessor and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    else:
        # No preprocessor, just use the model
        pipeline = model

    # Scoring metrics for cross-validation (using macro averaging)
    scoring = {
        'MAE': 'neg_mean_absolute_error',
        'RMSE': 'neg_root_mean_squared_error',
        'R²': 'r2'
    }

    # Perform cross-validation on training data
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False
    )

    metrics = {
        'MAE': -cv_results['test_MAE'].mean(),
        'RMSE': -cv_results['test_RMSE'].mean(),
        'R²': cv_results['test_R²'].mean()
    }

    # p.plot_classification_results(metrics, model_name)

    return metrics




def train_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, seed=None):
    # Set random seed if provided and model has the parameter
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)

    # Evaluate
    metrics = evaluate_regression(
        y_test=y_test,
        y_pred=y_pred
    )

    return metrics


def calculate_regression_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return mae, rmse, r2


def evaluate_regression(y_test, y_pred):
    mae, rmse, r2 = calculate_regression_metrics(y_test, y_pred)
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2
    }
    return metrics

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae, rmse, r2 = calculate_regression_metrics(y_test, y_pred)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

def heatmap_metrics(metrics_df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True)
    plt.title("Model Evaluation Metrics")
    plt.ylabel("Models")
    plt.xlabel("Metrics")
    plt.show()