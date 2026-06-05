from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate  # pretty console tables

def evaluate_model(true_values, predicted_values):
    """
    Compute evaluation metrics: MAE, RMSE, R².

    Parameters
    ----------
    true_values : array-like
        Ground truth values.
    predicted_values : array-like
        Predicted values by model.

    Returns
    -------
    dict
        Dictionary with MAE, RMSE, R².
    """
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    r2 = r2_score(true_values, predicted_values)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def compare_models(results_dict):
    """
    Compare multiple models by sorting results on RMSE.

    Parameters
    ----------
    results_dict : dict
        Example:
        {
            "IDW": {"MAE": ..., "RMSE": ..., "R2": ...},
            "Kriging": {"MAE": ..., "RMSE": ..., "R2": ...}
        }

    Returns
    -------
    pd.DataFrame
        Sorted dataframe of results.
    """
    df = pd.DataFrame(results_dict).T
    df_sorted = df.sort_values(by="RMSE")
    return df_sorted

def plot_evaluation(results_dict, method_name=""):
    """
    Plot bar chart of evaluation metrics for a single method.

    Parameters
    ----------
    results_dict : dict
        Dictionary with MAE, RMSE, R².
    method_name : str
        Name of the method for plot title.
    """
    metrics = list(results_dict.keys())
    values = list(results_dict.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(metrics, values, color=['orange', 'skyblue', 'green'])
    plt.title(f"Evaluation Metrics for {method_name}")
    plt.ylabel("Value")

    # annotate each bar with its value
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01,
                 f"{yval:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

def display_results_table(results_dict):
    """
    Display evaluation results of all models as a table in console.
    """
    df = compare_models(results_dict)
    df = df.round(2)
    df.index.name = "Method"
    df = df.reset_index()

    print("\n📊 Evaluation results of models:")
    print(tabulate(df, headers='keys', tablefmt='github', showindex=False))

def save_results_table(results_dict, filename="evaluation_table.txt"):
    """
    Save evaluation results as text file with markdown-style table.
    """
    df = compare_models(results_dict)
    df = df.round(2)
    df.index.name = "Method"
    df = df.reset_index()

    table_str = tabulate(df, headers='keys', tablefmt='github', showindex=False)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(table_str)
    print(f"✅ Results table saved to '{filename}'")

def plot_comparison_bar_chart(all_results):
    """
    Plot stacked bar chart comparing methods by RMSE and R².

    Parameters
    ----------
    all_results : list of dict
        Example:
        [
            {"Method": "IDW", "RMSE": ..., "R2": ...},
            {"Method": "Kriging", "RMSE": ..., "R2": ...}
        ]
    """
    methods = [r['Method'] for r in all_results]
    rmse = [r['RMSE'] for r in all_results]
    r2 = [r['R2'] for r in all_results]

    plt.figure(figsize=(10, 5))
    plt.bar(methods, rmse, label='RMSE')
    plt.bar(methods, r2, bottom=rmse, label='R²')
    plt.title("Comparison of Interpolation Methods")
    plt.legend()
    plt.tight_layout()
    plt.show()
