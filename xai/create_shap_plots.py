import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from utils.data import load_splits, rescale_data

# Configuration
config = {
    "input_dir": "xai/shap_analysis",
    "plots_dir": "plots/shap_analysis",
}


def load_shap_analysis_data(input_dir):
    """Load pre-computed SHAP analysis data"""
    print("=" * 80)
    print("LOADING SHAP ANALYSIS DATA")
    print("=" * 80)

    # Check if required files exist
    required_files = [
        "analysis_summary.json",
        "xgb_shap_values.npy",
        "gru_shap_values.npy",
        "explanation_features.csv",
        "expected_values.json",
        "sampling_manager_state.json",
    ]

    for file in required_files:
        file_path = os.path.join(input_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Load analysis summary
    with open(os.path.join(input_dir, "analysis_summary.json"), "r") as f:
        analysis_summary = json.load(f)

    # Load SHAP values
    xgb_shap_values = np.load(os.path.join(input_dir, "xgb_shap_values.npy"))
    gru_shap_values = np.load(os.path.join(input_dir, "gru_shap_values.npy"))

    # Load explanation features
    explanation_features = pd.read_csv(
        os.path.join(input_dir, "explanation_features.csv")
    )

    # Load expected values
    with open(os.path.join(input_dir, "expected_values.json"), "r") as f:
        expected_values = json.load(f)

    # Load sampling manager state
    with open(os.path.join(input_dir, "sampling_manager_state.json"), "r") as f:
        manager_state = json.load(f)

    print(f"✓ Analysis summary loaded")
    print(f"✓ XGBoost SHAP values loaded: {xgb_shap_values.shape}")
    print(f"✓ GRU SHAP values loaded: {gru_shap_values.shape}")
    print(f"✓ Explanation features loaded: {explanation_features.shape}")
    print(f"✓ Expected values loaded")
    print(f"✓ Sampling manager state loaded")

    return {
        "analysis_summary": analysis_summary,
        "xgb_shap_values": xgb_shap_values,
        "gru_shap_values": gru_shap_values,
        "explanation_features": explanation_features,
        "expected_values": expected_values,
        "manager_state": manager_state,
        "X_cols": analysis_summary["X_cols"],
    }


def rescale_shap_data(data):
    """Rescale SHAP values, features, and expected values back to original range"""
    print("\n" + "=" * 80)
    print("RESCALING DATA TO ORIGINAL VALUES")
    print("=" * 80)

    # Constants for rescaling (from utils.data)
    L_BOUND = 40.0
    U_BOUND = 400.0

    # Calculate scaling factor for SHAP values
    # SHAP values need to be rescaled by the derivative of the rescaling function
    # Original: value = ((scaled + 1) * (U_BOUND - L_BOUND) / 2) + L_BOUND
    # Derivative: d(value)/d(scaled) = (U_BOUND - L_BOUND) / 2
    scaling_factor = (U_BOUND - L_BOUND) / 2

    print(f"Scaling factor for SHAP values: {scaling_factor}")

    # Rescale features (lag values)
    explanation_features_rescaled = data["explanation_features"].copy()
    for col in data["X_cols"]:
        explanation_features_rescaled[col] = (
            (explanation_features_rescaled[col] + 1) * (U_BOUND - L_BOUND) / 2
        ) + L_BOUND

    print(f"✓ Features rescaled from [-1, 1] to [{L_BOUND}, {U_BOUND}] mg/dL")

    # Rescale SHAP values
    xgb_shap_values_rescaled = data["xgb_shap_values"] * scaling_factor
    gru_shap_values_rescaled = data["gru_shap_values"] * scaling_factor

    print(f"✓ SHAP values rescaled by factor {scaling_factor}")

    # Rescale expected values
    expected_values_rescaled = {}
    for key, value in data["expected_values"].items():
        expected_values_rescaled[key] = (
            (value + 1) * (U_BOUND - L_BOUND) / 2
        ) + L_BOUND

    print(f"✓ Expected values rescaled:")
    print(
        f"   XGBoost: {data['expected_values']['xgb_expected_value']:.4f} → {expected_values_rescaled['xgb_expected_value']:.4f}"
    )
    print(
        f"   GRU: {data['expected_values']['gru_expected_value']:.4f} → {expected_values_rescaled['gru_expected_value']:.4f}"
    )

    # Update data dictionary with rescaled values
    data_rescaled = data.copy()
    data_rescaled.update(
        {
            "explanation_features_rescaled": explanation_features_rescaled,
            "xgb_shap_values_rescaled": xgb_shap_values_rescaled,
            "gru_shap_values_rescaled": gru_shap_values_rescaled,
            "expected_values_rescaled": expected_values_rescaled,
        }
    )

    return data_rescaled


def create_summary_plots(data, output_dir):
    """Create summary plots for both models with rescaled values"""
    print("\n" + "=" * 80)
    print("CREATING SUMMARY PLOTS (RESCALED VALUES)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    X_cols = data["X_cols"]
    explanation_features_rescaled = data["explanation_features_rescaled"].values

    # XGBoost Summary Plot
    print("Creating XGBoost summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        data["xgb_shap_values_rescaled"],
        explanation_features_rescaled,
        feature_names=X_cols,
        show=False,
    )
    plt.title("XGBoost - SHAP Summary Plot (Rescaled Values)", fontsize=14, pad=20)
    plt.xlabel("SHAP value (impact on model output in mg/dL)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/xgb_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # GRU Summary Plot
    print("Creating GRU summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        data["gru_shap_values_rescaled"],
        explanation_features_rescaled,
        feature_names=X_cols,
        show=False,
    )
    plt.title("GRU - SHAP Summary Plot (Rescaled Values)", fontsize=14, pad=20)
    plt.xlabel("SHAP value (impact on model output in mg/dL)", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gru_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Summary plots saved to {output_dir}/")
    print(f"   - Features displayed in original glucose units (mg/dL)")
    print(f"   - SHAP values show impact in mg/dL")


def create_waterfall_plots(data, output_dir):
    """Create waterfall plots for clinical cases with rescaled values"""
    print("\n" + "=" * 80)
    print("CREATING WATERFALL PLOTS (RESCALED VALUES)")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Load test set to find clinical cases
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    X_cols = data["X_cols"]
    explanation_features_rescaled = data["explanation_features_rescaled"].values
    manager_state = data["manager_state"]
    expected_values_rescaled = data["expected_values_rescaled"]

    # Recreate explanation data from indices
    explanation_indices = manager_state["explanation_indices"]
    explanation_data = test_set.iloc[explanation_indices].copy()

    # Find corresponding indices in explanation data for clinical cases
    clinical_indices = {}

    for bg_class, case_info in manager_state["clinical_cases"].items():
        # Find the index in explanation data that corresponds to this clinical case
        original_idx = case_info["index"]

        # Check if this case is in our explanation data
        explanation_mask = explanation_data.index == original_idx
        if explanation_mask.any():
            explanation_idx = np.where(explanation_mask)[0][0]
            clinical_indices[bg_class] = explanation_idx

            # Get rescaled target value for display
            target_value_rescaled = (
                (case_info["target_value"] + 1) * (400 - 40) / 2
            ) + 40
            print(f"✓ {bg_class} case found at explanation index {explanation_idx}")
            print(f"   Target value: {target_value_rescaled:.1f} mg/dL")
        else:
            print(f"⚠️  Warning: {bg_class} case not found in explanation data")

    # Create waterfall plots for each clinical case
    for bg_class, exp_idx in clinical_indices.items():
        if bg_class.lower() in ["hypo", "normal", "hyper"]:

            print(f"\nCreating waterfall plots for {bg_class} case...")

            # XGBoost waterfall plot
            if exp_idx < len(data["xgb_shap_values_rescaled"]):
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=data["xgb_shap_values_rescaled"][exp_idx],
                        base_values=expected_values_rescaled["xgb_expected_value"],
                        data=explanation_features_rescaled[exp_idx],
                        feature_names=X_cols,
                    ),
                    show=False,
                )
                plt.title(
                    f"XGBoost - {bg_class} Case Waterfall Plot\n(Values in mg/dL)",
                    fontsize=12,
                )
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/xgb_waterfall_{bg_class.lower()}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

            # GRU waterfall plot
            if exp_idx < len(data["gru_shap_values_rescaled"]):
                plt.figure(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=data["gru_shap_values_rescaled"][exp_idx],
                        base_values=expected_values_rescaled["gru_expected_value"],
                        data=explanation_features_rescaled[exp_idx],
                        feature_names=X_cols,
                    ),
                    show=False,
                )
                plt.title(
                    f"GRU - {bg_class} Case Waterfall Plot\n(Values in mg/dL)",
                    fontsize=12,
                )
                plt.tight_layout()
                plt.savefig(
                    f"{output_dir}/gru_waterfall_{bg_class.lower()}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    print(f"✓ Waterfall plots saved to {output_dir}/")
    print(f"   - All values displayed in original glucose units (mg/dL)")
    print(f"   - SHAP values show contribution in mg/dL")


def create_feature_importance_comparison(data, output_dir):
    """Create feature importance comparison between models with rescaled values"""
    print("\n" + "=" * 80)
    print("CREATING FEATURE IMPORTANCE COMPARISON (RESCALED VALUES)")
    print("=" * 80)

    X_cols = data["X_cols"]

    # Calculate mean absolute SHAP values for each feature (using rescaled values)
    xgb_importance = np.abs(data["xgb_shap_values_rescaled"]).mean(axis=0)
    gru_importance = np.abs(data["gru_shap_values_rescaled"]).mean(axis=0)

    # Create comparison DataFrame
    importance_df = pd.DataFrame(
        {"Feature": X_cols, "XGBoost": xgb_importance, "GRU": gru_importance}
    )

    # Create comparison plot
    plt.figure(figsize=(12, 8))
    x_pos = np.arange(len(X_cols))
    width = 0.35

    plt.bar(
        x_pos - width / 2,
        importance_df["XGBoost"],
        width,
        label="XGBoost",
        alpha=0.8,
        color="skyblue",
    )
    plt.bar(
        x_pos + width / 2,
        importance_df["GRU"],
        width,
        label="GRU",
        alpha=0.8,
        color="lightcoral",
    )

    plt.xlabel("Features")
    plt.ylabel("Mean |SHAP Value| (mg/dL)")
    plt.title(
        "Feature Importance Comparison: XGBoost vs GRU\n(Impact on glucose prediction in mg/dL)"
    )
    plt.xticks(x_pos, X_cols, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/feature_importance_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save importance values
    importance_df.to_csv(f"{output_dir}/feature_importance_comparison.csv", index=False)

    print(f"✓ Feature importance comparison saved to {output_dir}/")
    print(f"   - Importance values represent average impact in mg/dL")

    # Print top features for each model
    print(f"\nTop 3 features by importance:")
    print(f"XGBoost:")
    for i, (idx, row) in enumerate(importance_df.nlargest(3, "XGBoost").iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['XGBoost']:.2f} mg/dL")

    print(f"GRU:")
    for i, (idx, row) in enumerate(importance_df.nlargest(3, "GRU").iterrows()):
        print(f"  {i+1}. {row['Feature']}: {row['GRU']:.2f} mg/dL")


def main():
    """Main function to create SHAP visualizations with rescaled values"""
    print("=" * 80)
    print("SHAP VISUALIZATION GENERATION")
    print("Creating plots from pre-computed SHAP values (with rescaling)")
    print("=" * 80)

    # Create output directory
    os.makedirs(config["plots_dir"], exist_ok=True)

    # Load SHAP analysis data
    data = load_shap_analysis_data(config["input_dir"])

    # Rescale data to original glucose units
    data_rescaled = rescale_shap_data(data)

    # Create visualizations with rescaled values
    create_summary_plots(data_rescaled, config["plots_dir"])
    create_waterfall_plots(data_rescaled, config["plots_dir"])
    create_feature_importance_comparison(data_rescaled, config["plots_dir"])

    print("\n" + "=" * 80)
    print("SHAP VISUALIZATION GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"All plots saved in: {config['plots_dir']}")
    print(f"\nGenerated files (all with rescaled values in mg/dL):")
    print(f"  📊 Summary plots:")
    print(f"     - xgb_summary_plot.png")
    print(f"     - gru_summary_plot.png")
    print(f"  💧 Waterfall plots:")
    print(f"     - xgb_waterfall_hypo.png")
    print(f"     - xgb_waterfall_normal.png")
    print(f"     - xgb_waterfall_hyper.png")
    print(f"     - gru_waterfall_hypo.png")
    print(f"     - gru_waterfall_normal.png")
    print(f"     - gru_waterfall_hyper.png")
    print(f"  📈 Feature importance:")
    print(f"     - feature_importance_comparison.png")
    print(f"     - feature_importance_comparison.csv")
    print(f"\n✨ All values are now displayed in original glucose units (mg/dL)")
    print(f"✨ SHAP values represent the impact on glucose prediction in mg/dL")


if __name__ == "__main__":
    main()
