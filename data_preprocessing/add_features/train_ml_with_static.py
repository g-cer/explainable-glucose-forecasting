import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data import load_splits
from training.ml.train_std_ml import create_model, save_model, evaluate_and_save_results


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs/val_set")
    parser.add_argument("--models_path", type=str, default="models/val_set")
    parser.add_argument("--plots_path", type=str, default="plots/val_set")
    parser.add_argument(
        "--exp_name",
        type=str,
        choices=["lgb", "xgb", "rf"],
        # required=True,
        default="xgb",
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def analyze_feature_importance(model, X_cols, plots_dir):
    """Analyze and visualize feature importance."""
    print("📈 Analyzing feature importance...")

    # Get feature importance
    importance_gain = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": X_cols, "importance": importance_gain}
    ).sort_values("importance", ascending=False)

    # Categorize features
    def categorize_feature(feature):
        if feature.startswith("lag"):
            return "Lag Glicemiche"
        elif feature in ["Sex", "Age"]:
            return "Demografiche"
        elif feature in ["HbA1c", "TSH", "Creatinine", "HDL", "Triglycerides"]:
            return "Biochimiche"
        else:
            return "Other"

    importance_df["category"] = importance_df["feature"].apply(categorize_feature)

    # Print top features
    print("\n🏆 Top 10 Most Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(
            f"   {row['feature']:<15} ({row['category']:<12}): {row['importance']:.4f}"
        )

    # Summary by category
    category_importance = importance_df.groupby("category")["importance"].agg(
        ["sum", "mean", "count"]
    )
    print(f"\n📊 Feature Importance by Category:")
    for category, stats in category_importance.iterrows():
        print(
            f"   {category:<12}: Total={stats['sum']:.4f}, Mean={stats['mean']:.4f}, Count={stats['count']}"
        )

    return importance_df, category_importance


def create_feature_importance_plots(importance_df, category_importance, plots_dir):
    """Create and save feature importance visualizations."""
    print("🎨 Creating feature importance visualizations...")

    os.makedirs(plots_dir, exist_ok=True)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create category comparison plot with closer bars
    fig2, ax = plt.subplots(
        figsize=(9.5, 3)
    )  # Made slightly smaller for better proportion

    # Prepare data for categorical comparison
    categories = ["Demografiche", "Biochimiche", "Lag Glicemiche"]
    category_totals = []
    # Use softer, more muted colors
    category_colors = [
        "#90C695",
        "#F4A460",
        "#87CEEB",
    ]  # Softer green, sand brown, light blue

    for cat in categories:
        if cat in category_importance.index:
            category_totals.append(category_importance.loc[cat, "sum"])
        else:
            category_totals.append(0)

    # Create normalized values for visual clarity (not maintaining real proportions)
    # Use log scale or square root to compress the range
    import numpy as np

    normalized_values = np.sqrt(category_totals)  # Square root normalization

    # Create positions for bars to be closer together
    bar_positions = np.arange(len(categories))

    # Create horizontal bar chart with thinner bars and closer spacing
    bars = ax.barh(
        bar_positions,
        normalized_values,
        color=category_colors,
        alpha=0.8,  # Slightly increased alpha for better visibility with softer colors
        edgecolor="black",  # Changed to gray for softer appearance
        linewidth=0.8,  # Reduced line width
        # height=0.35,  # Increased height to make bars thicker and closer
    )

    # Set y-axis properties for closer spacing
    ax.set_yticks(bar_positions)
    # ax.set_ylim(-0.4, len(categories) - 0.6)  # Reduced margins to bring bars closer

    ax.set_xlim(0, max(normalized_values) * 1.15)  # Adjusted x-limit
    ax.set_yticklabels(categories, fontsize=14)  # Increased font size

    ax.set_xlabel(
        "Feature Importance (Normalizzata per chiarezza visiva)", fontsize=14
    )  # Increased font size
    # ax.set_xlabel("", fontsize=14)  # Increased font size
    ax.grid(True, alpha=0.2, axis="x", color="lightgray")  # Softer grid

    # Increase tick label font size
    ax.tick_params(axis="x", labelsize=12)

    # Add actual values and percentages on bars with larger font
    total_importance = sum(category_totals)
    for i, (bar, actual_val, norm_val) in enumerate(
        zip(bars, category_totals, normalized_values)
    ):
        percentage = (
            (actual_val / total_importance * 100) if total_importance > 0 else 0
        )
        ax.text(
            norm_val + max(normalized_values) * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"({percentage:.1f}%)",
            va="center",
            # fontweight="bold",
            fontsize=14,  # Increased from 11 to 14
            # color="darkslategray",  # Softer text color
        )

    plt.tight_layout()

    # Save category comparison plot as both PNG and PDF
    category_plot_path_png = f"{plots_dir}/xgb_category_importance_comparison.png"
    category_plot_path_pdf = f"{plots_dir}/xgb_category_importance_comparison.pdf"

    plt.savefig(category_plot_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(category_plot_path_pdf, dpi=300, bbox_inches="tight", format="pdf")
    plt.show()

    print(f"✓ Category comparison plot saved to: {category_plot_path_png}")
    print(f"✓ Category comparison plot (PDF) saved to: {category_plot_path_pdf}")

    # Print summary statistics
    print(f"\n📊 Category Importance Summary:")
    for cat, total, color in zip(categories, category_totals, category_colors):
        percentage = (total / total_importance * 100) if total_importance > 0 else 0
        count = (
            category_importance.loc[cat, "count"]
            if cat in category_importance.index
            else 0
        )
        print(f"   • {cat:<15}: {total:.4f} ({percentage:.1f}%) - {count} features")


def main():
    """Main training function"""
    args = parse_arguments()

    print(f"Starting {args.exp_name.upper()} training pipeline...")

    # Setup
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.models_path, exist_ok=True)
    os.makedirs(args.plots_path, exist_ok=True)

    # Load data
    print("Loading pre-prepared static data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits("data/static_split_sets")

    # Create and train model
    print(f"Creating and training {args.exp_name.upper()} model...")
    model = create_model(args.exp_name, args.seed)
    model.fit(train_set[X_cols], train_set[y_cols[-1]])

    # Save model
    model_path = f"{args.models_path}/{args.exp_name}_static.pickle"
    save_model(model, model_path)

    print()  # Add spacing

    # Evaluate and save results
    results = evaluate_and_save_results(
        model, val_set, X_cols, y_cols, args.output_path, args.exp_name + "_static"
    )

    # Analyze feature importance
    importance_df, category_importance = analyze_feature_importance(
        model, X_cols, args.plots_path
    )

    # Create visualizations
    create_feature_importance_plots(importance_df, category_importance, args.plots_path)

    print(f"\n🎉 XGBoost training with static features completed successfully!")
    print(f"\n📊 Key Insights:")
    glucose_importance = (
        category_importance.loc["Lag Glicemiche", "sum"]
        if "Lag Glicemiche" in category_importance.index
        else 0
    )
    biochemical_importance = (
        category_importance.loc["Biochimiche", "sum"]
        if "Biochimiche" in category_importance.index
        else 0
    )
    total_importance = category_importance["sum"].sum()

    print(
        f"   • Glucose lag features contribute {glucose_importance/total_importance*100:.1f}% of total importance"
    )
    print(
        f"   • Biochimiche features contribute {biochemical_importance/total_importance*100:.1f}% of total importance"
    )
    print(
        f"   • Top biochemical feature: {importance_df[importance_df['category']=='Biochimiche'].iloc[0]['feature'] if not importance_df[importance_df['category']=='Biochimiche'].empty else 'None'}"
    )

    return model, results, importance_df


if __name__ == "__main__":
    model, results, importance_df = main()
