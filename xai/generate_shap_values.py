import os
import pickle
import warnings
import numpy as np
import pandas as pd
import shap

from utils.data import load_splits, rescale_data, categorize_glucose
from utils.tf_dnn import create_gru_model, predict_in_batches

# Configuration
config = {
    "xgb_model_path": "models/test_set/xgb.pickle",
    "gru_model_path": "models/val_set/gru.weights.h5",
    "test_results_xgb": "outputs/test_set/xgb_output.csv",
    "test_results_gru": "outputs/test_set/gru_output.csv",
    "output_dir": "xai/shap_analysis",
    "plots_dir": "plots/shap_analysis",
    "background_size": 50,  # Number of background samples for SHAP
    "explanation_size": 250,  # Number of samples to explain
    "random_seed": 42,
}


class SharedSamplingManager:
    """Manages consistent sampling across XGBoost and GRU analyses"""

    def __init__(
        self,
        test_set,
        train_set,
        background_size=100,
        explanation_size=500,
        random_seed=42,
    ):
        self.test_set = test_set.copy()
        self.train_set = train_set.copy()
        self.background_size = background_size
        self.explanation_size = explanation_size
        self.random_seed = random_seed

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Create consistent samples
        self._create_background_samples()
        self._create_explanation_samples()
        self._identify_clinical_cases()

    def _create_background_samples(self):
        """Create background samples for SHAP explainers"""
        print(f"Creating {self.background_size} background samples...")

        # Sample randomly from train set
        self.background_indices = np.random.choice(
            len(self.train_set), size=self.background_size, replace=False
        )
        self.background_data = self.train_set.iloc[self.background_indices].copy()

        print(f"Background samples by class:")
        print(self.background_data["bgClass"].value_counts())

    def _create_explanation_samples(self):
        """Create explanation samples for SHAP analysis"""
        print(f"\nCreating {self.explanation_size} explanation samples...")

        # Sample randomly from test set
        self.explanation_indices = np.random.choice(
            len(self.train_set), size=self.explanation_size, replace=False
        )
        self.explanation_data = self.train_set.iloc[self.explanation_indices].copy()

        print(f"Explanation samples by class:")
        print(self.explanation_data["bgClass"].value_counts())

    def _identify_clinical_cases(self):
        """Identify specific instances for each clinical case"""
        print(f"\nIdentifying clinical cases...")

        self.clinical_cases = {}

        # Find representative instances for each class
        for bg_class in ["Hypo", "Normal", "Hyper"]:
            class_data = self.explanation_data[
                self.explanation_data["bgClass"] == bg_class
            ]

            if len(class_data) > 0:
                # Choose instance closest to median target value for the class
                median_target = class_data["lead30"].median()
                distances = np.abs(class_data["lead30"] - median_target)
                best_idx = distances.idxmin()

                self.clinical_cases[bg_class] = {
                    "index": best_idx,
                    "data": class_data.loc[best_idx],
                    "target_value": class_data.loc[best_idx, "lead30"],
                }

                print(
                    f"{bg_class}: Target = {self.clinical_cases[bg_class]['target_value']:.1f}"
                )
            else:
                print(f"Warning: No {bg_class} samples found in explanation data")

        return self.clinical_cases


def load_xgb_model():
    """Load the trained XGBoost model"""
    print("\n" + "=" * 80)
    print("LOADING XGBOOST MODEL")
    print("=" * 80)

    if not os.path.exists(config["xgb_model_path"]):
        raise FileNotFoundError(
            f"XGBoost model not found at {config['xgb_model_path']}"
        )

    with open(config["xgb_model_path"], "rb") as f:
        model = pickle.load(f)

    print(f"XGBoost model loaded from {config['xgb_model_path']}")
    print(f"Model type: {type(model)}")

    if hasattr(model, "n_estimators"):
        print(f"Number of estimators: {model.n_estimators}")
    if hasattr(model, "max_depth"):
        print(f"Max depth: {model.max_depth}")

    return model


def load_gru_model():
    """Load the trained GRU model"""
    print("\n" + "=" * 80)
    print("LOADING GRU MODEL")
    print("=" * 80)

    if not os.path.exists(config["gru_model_path"]):
        raise FileNotFoundError(f"GRU model not found at {config['gru_model_path']}")

    # Create model with same architecture as training
    model = create_gru_model()
    model.load_weights(config["gru_model_path"])

    print(f"GRU model loaded from {config['gru_model_path']}")
    print(f"Model architecture:")
    model.summary()

    return model


def create_gru_wrapper(gru_model, X_cols):
    """Create a wrapper function for GRU model to work with SHAP"""

    def predict_wrapper(X):
        """Wrapper function that reshapes input and returns predictions"""
        # X is expected to be 2D (n_samples, n_features)
        # Reshape to 3D for GRU: (n_samples, n_features, 1)
        if len(X.shape) == 2:
            X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
        else:
            X_reshaped = X

        # Make predictions
        predictions = gru_model.predict(X_reshaped, verbose=0)

        # Return flattened predictions
        return predictions.flatten()

    return predict_wrapper


def analyze_xgb_shap(xgb_model, manager, X_cols):
    """Perform SHAP analysis for XGBoost model"""
    print("\n" + "=" * 80)
    print("XGBOOST SHAP ANALYSIS")
    print("=" * 80)

    # Prepare data for XGBoost (2D format)
    X_background = manager.background_data[X_cols].values
    X_explanation = manager.explanation_data[X_cols].values

    print(f"Background data shape: {X_background.shape}")
    print(f"Explanation data shape: {X_explanation.shape}")

    # Create TreeExplainer (optimal for tree-based models)
    print("Creating TreeExplainer for XGBoost...")
    explainer = shap.TreeExplainer(xgb_model, X_background)

    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = explainer.shap_values(X_explanation)

    print(f"SHAP values calculated successfully!")
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Expected value (baseline): {explainer.expected_value:.4f}")

    return explainer, shap_values, X_explanation


def analyze_gru_shap(gru_model, manager, X_cols):
    """Perform SHAP analysis for GRU model"""
    print("\n" + "=" * 80)
    print("GRU SHAP ANALYSIS")
    print("=" * 80)

    # Prepare data for GRU (2D format for SHAP, will be reshaped in wrapper)
    X_background = manager.background_data[X_cols].values
    X_explanation = manager.explanation_data[X_cols].values

    print(f"Background data shape: {X_background.shape}")
    print(f"Explanation data shape: {X_explanation.shape}")

    # Create wrapper function for GRU
    gru_wrapper = create_gru_wrapper(gru_model, X_cols)

    # Create KernelExplainer with reduced background size for efficiency
    print("Creating KernelExplainer for GRU...")
    explainer = shap.KernelExplainer(gru_wrapper, X_background)

    # Calculate SHAP values for a subset of explanation data
    print("Calculating SHAP values (this may take a while for KernelExplainer)...")
    shap_values = explainer.shap_values(X_explanation, nsamples=512)

    print(f"SHAP values calculated successfully!")
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Expected value (baseline): {explainer.expected_value:.4f}")

    return explainer, shap_values, X_explanation


def save_shap_analysis(xgb_results, gru_results, manager, X_cols, output_dir):
    """Save SHAP analysis results and metadata"""
    print("\n" + "=" * 80)
    print("SAVING SHAP ANALYSIS RESULTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Create summary report
    summary = {
        "analysis_config": config,
        "X_cols": X_cols,
        "background_samples": len(manager.background_data),
        "explanation_samples": len(manager.explanation_data),
        "background_class_distribution": manager.background_data["bgClass"]
        .value_counts()
        .to_dict(),
        "explanation_class_distribution": manager.explanation_data["bgClass"]
        .value_counts()
        .to_dict(),
        "clinical_cases": {},
    }

    # Add clinical cases info
    for bg_class, case_info in manager.clinical_cases.items():
        summary["clinical_cases"][bg_class] = {
            "target_value": float(case_info["target_value"]),
            "index": int(case_info["index"]),
        }

    # Add model-specific info
    summary["xgb_analysis"] = {
        "shap_values_shape": list(xgb_results["shap_values"].shape),
        "expected_value": float(xgb_results["explainer"].expected_value),
        "explainer_type": "TreeExplainer",
    }

    summary["gru_analysis"] = {
        "shap_values_shape": list(gru_results["shap_values"].shape),
        "expected_value": float(gru_results["explainer"].expected_value),
        "explainer_type": "KernelExplainer",
    }

    # Save summary as JSON
    import json

    with open(f"{output_dir}/analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save SHAP values as numpy arrays for future use
    np.save(f"{output_dir}/xgb_shap_values.npy", xgb_results["shap_values"])
    np.save(f"{output_dir}/gru_shap_values.npy", gru_results["shap_values"])

    # Save explanation data
    explanation_features = pd.DataFrame(
        xgb_results["X_explanation"],
        columns=X_cols,
    )
    explanation_features.to_csv(f"{output_dir}/explanation_features.csv", index=False)

    # Save background data
    background_features = pd.DataFrame(
        manager.background_data[X_cols].values,
        columns=X_cols,
    )
    background_features.to_csv(f"{output_dir}/background_features.csv", index=False)

    # Save expected values separately
    expected_values = {
        "xgb_expected_value": float(xgb_results["explainer"].expected_value),
        "gru_expected_value": float(gru_results["explainer"].expected_value),
    }
    with open(f"{output_dir}/expected_values.json", "w") as f:
        json.dump(expected_values, f, indent=2)

    # Save sampling manager state for visualization script
    manager_state = {
        "background_indices": manager.background_indices.tolist(),
        "explanation_indices": manager.explanation_indices.tolist(),
        "clinical_cases": {},
    }

    for bg_class, case_info in manager.clinical_cases.items():
        manager_state["clinical_cases"][bg_class] = {
            "index": int(case_info["index"]),
            "target_value": float(case_info["target_value"]),
        }

    with open(f"{output_dir}/sampling_manager_state.json", "w") as f:
        json.dump(manager_state, f, indent=2)

    print(f"SHAP analysis results saved to {output_dir}/")
    print(f"Files created:")
    print(f"  - analysis_summary.json")
    print(f"  - xgb_shap_values.npy")
    print(f"  - gru_shap_values.npy")
    print(f"  - explanation_features.csv")
    print(f"  - background_features.csv")
    print(f"  - expected_values.json")
    print(f"  - sampling_manager_state.json")


def main():
    """Main function to generate SHAP values"""
    print("=" * 80)
    print("SHAP VALUES GENERATION")
    print("XGBoost (TreeExplainer) vs GRU (KernelExplainer)")
    print("=" * 80)

    # Create output directories
    os.makedirs(config["output_dir"], exist_ok=True)

    # Load data
    print("\nLoading data...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    print(f"Dataset sizes:")
    print(f"  Test set: {len(test_set)} samples")
    print(f"  Features: {X_cols}")
    print(f"  Target: {y_cols}")

    # Create shared sampling manager
    print("\nInitializing shared sampling manager...")
    manager = SharedSamplingManager(
        test_set,
        train_set,
        background_size=config["background_size"],
        explanation_size=config["explanation_size"],
        random_seed=config["random_seed"],
    )

    # Load models
    xgb_model = load_xgb_model()
    gru_model = load_gru_model()

    # Perform SHAP analysis
    print("\nPerforming SHAP analyses...")

    # XGBoost Analysis
    xgb_explainer, xgb_shap_values, xgb_X_explanation = analyze_xgb_shap(
        xgb_model, manager, X_cols
    )

    # GRU Analysis
    gru_explainer, gru_shap_values, gru_X_explanation = analyze_gru_shap(
        gru_model, manager, X_cols
    )

    # Package results
    xgb_results = {
        "explainer": xgb_explainer,
        "shap_values": xgb_shap_values,
        "X_explanation": xgb_X_explanation,
    }

    gru_results = {
        "explainer": gru_explainer,
        "shap_values": gru_shap_values,
        "X_explanation": gru_X_explanation,
    }

    # Save analysis results
    save_shap_analysis(xgb_results, gru_results, manager, X_cols, config["output_dir"])

    print("\n" + "=" * 80)
    print("SHAP VALUES GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Analysis data saved in: {config['output_dir']}")
    print(f"\nGenerated files:")
    print(f"  💾 SHAP values: xgb_shap_values.npy, gru_shap_values.npy")
    print(f"  📋 Analysis summary: analysis_summary.json")
    print(f"  📊 Feature data: explanation_features.csv, background_features.csv")
    print(f"  🎯 Expected values: expected_values.json")
    print(f"  🔄 Sampling state: sampling_manager_state.json")
    print(f"\nNext step: Run 'python create_shap_plots.py' to generate visualizations.")


if __name__ == "__main__":
    main()
