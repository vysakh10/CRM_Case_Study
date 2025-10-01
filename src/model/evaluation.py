from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def precision_recall_at_k(
    y_true: pd.Series | list, y_probs: pd.Series | list, k_percent: float = 0.1
) -> Tuple[float, float]:
    """
    Compute precision and recall @ top K% predicted probabilities.
    Args:
        y_true (pd.Series or list): True labels (0/1)
        y_probs (pd.Series or list): Predicted probabilities
        k_percent (float): Top fraction to consider (e.g., 0.1 for top 10%)
    Returns:
        Tuple[float, float]: precision, recall
    """
    # Create a DataFrame
    df = pd.DataFrame({"y_true": y_true, "y_prob": y_probs})

    # Sort by predicted probability descending
    df = df.sort_values("y_prob", ascending=False)

    # Select top K%
    top_k_count = max(1, int(len(df) * k_percent))
    top_k = df.head(top_k_count)
    # Compute precision and recall
    precision = top_k["y_true"].sum() / len(top_k)
    recall = top_k["y_true"].sum() / df["y_true"].sum()

    return precision, recall


def plot_top_k_metrics(
    y_true: pd.Series | list,
    y_probs: pd.Series | list,
    set: str = "Test",
    top_percent: int = 20,
    save_path: str = None,
) -> None:
    """
    Plots a heatmap of precision and recall at each top K% from 1 to top_percent.
    Args:
        y_true (pd.Series or list): True labels (0/1)
        y_probs (pd.Series or list): Predicted probabilities
        set (str, optional): Label for the dataset (e.g., 'Train', 'Val' and 'Test'). Defaults to "Test".
        top_percent (int, optional): Maximum K% to plot. Defaults to 20.
    Returns:
        None
    """
    metrics = []
    for k in range(1, top_percent + 1):
        precision, recall = precision_recall_at_k(y_true, y_probs, k / 100)
        metrics.append({"k%": k, "Precision": precision, "Recall": recall})

    metrics_df = pd.DataFrame(metrics)

    # Convert to long format for heatmap
    metrics_melt = metrics_df.melt(
        id_vars="k%",
        value_vars=["Precision", "Recall"],
        var_name="Metric",
        value_name="Score",
    )

    # Pivot for heatmap
    heatmap_data = metrics_melt.pivot(index="Metric", columns="k%", values="Score")

    plt.figure(figsize=(20, 4))
    sns.heatmap(
        heatmap_data, annot=True, cmap="Blues", fmt=".2f", cbar_kws={"label": "Score"}
    )
    plt.title(f"{set} Precision & Recall @ Top K% (1–{top_percent}%)")
    plt.xlabel("Top K%")
    plt.ylabel("Metric")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_importance(
    x_train: pd.DataFrame, model, save_path: str = None
) -> None:
    """
    Plots feature importances for a trained model.
    Args:
        x_train (pd.DataFrame): Training features with column names.
        model: Trained model with feature_importances_ or coef_ attribute.
    Returns:
        None
    """

    if type(model).__name__ == "SGDClassifier":
        feature_importance = model.coef_.ravel()
    else:
        feature_importance = model.feature_importances_

    # Map to feature names (make sure you pass X_train with column names when training)
    features = x_train.columns

    # Create a DataFrame
    feat_imp_df = pd.DataFrame(
        {"Feature": features, "Importance": feature_importance}
    ).sort_values(by="Importance", ascending=False)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df, palette="crest")
    plt.title(f"{type(model).__name__} Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
