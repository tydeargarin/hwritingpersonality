"""
Main Application / Backend API for Handwriting-Based Big Five Personality Analysis

Based on:
  "A Framework for Determining the Big Five Personality Traits Using
   Machine Learning Classification through Graphology"

Backend capabilities:
- Preprocess the handwriting dataset (utility functions)
- Extract handwriting features
- Map Big Five personality traits (rule-based)
- Train & evaluate SVM, KNN, and Decision Tree classifiers
- Analyze a single uploaded handwriting image end‑to‑end via a REST API
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from preprocess_handwriting import HandwritingPreprocessor
from extract_features import HandwritingFeatureExtractor
from map_personality_traits import PersonalityTraitMapper
from classify_personality_traits import PersonalityTraitClassifier


def run_preprocessing(input_dir: str = "handwriting",
                      output_dir: str = "handwriting_preprocessed") -> None:
    """Run preprocessing on the whole dataset."""
    print(f"\n[1/4] Preprocessing handwriting images: {input_dir} -> {output_dir}")
    preprocessor = HandwritingPreprocessor()
    preprocessor.preprocess_dataset(input_dir=input_dir, output_dir=output_dir)


def run_feature_extraction(input_dir: str = "handwriting",
                           preprocessed_dir: str = "handwriting_preprocessed",
                           output_csv: str = "handwriting_features.csv") -> None:
    """Run feature extraction on the dataset."""
    print(f"\n[2/4] Extracting handwriting features to {output_csv}")
    extractor = HandwritingFeatureExtractor()
    # Use preprocessed images if directory exists
    use_preprocessed = Path(preprocessed_dir).exists()
    if use_preprocessed:
        print(f"Using preprocessed images from: {preprocessed_dir}")
    extractor.extract_features_from_dataset(
        input_dir=input_dir,
        output_csv=output_csv,
        use_preprocessed=use_preprocessed,
        preprocessed_dir=preprocessed_dir if use_preprocessed else None,
    )


def run_trait_mapping(features_csv: str = "handwriting_features.csv",
                      output_csv: str = "personality_traits.csv") -> None:
    """Map handwriting features to Big Five traits for the whole dataset."""
    print(f"\n[3/4] Mapping handwriting features to Big Five traits")
    print(f"Loading features from {features_csv}...")
    features_df = pd.read_csv(features_csv)
    print(f"Loaded {len(features_df)} samples")

    mapper = PersonalityTraitMapper()
    predictions_df = mapper.predict_traits(features_df, use_classifier=False)

    if "image_name" in features_df.columns:
        predictions_df.insert(0, "image_name", features_df["image_name"].values)

    predictions_df.to_csv(output_csv, index=False)
    print(f"Personality trait scores saved to {output_csv}")


def run_classification(features_csv: str = "handwriting_features.csv",
                       labels_csv: str = "personality_traits.csv",
                       output_dir: str = "classification_results",
                       model_file: str = "personality_classifiers.pkl",
                       test_size: float = 0.2,
                       random_state: int = 42,
                       generate_plots: bool = True) -> None:
    """Train and evaluate SVM, KNN, and Decision Tree classifiers."""
    print(f"\n[4/4] Training and evaluating classifiers (SVM, KNN, Decision Tree)")

    print(f"Loading features from {features_csv}...")
    features_df = pd.read_csv(features_csv)
    feature_columns = features_df.columns.tolist()
    if "image_name" in features_df.columns:
        features_df = features_df.drop_duplicates(subset="image_name", keep="first")
        print(f"After removing duplicate images in features: {len(features_df)} samples")

    print(f"\nLoading labels from {labels_csv}...")
    labels_df = pd.read_csv(labels_csv)
    if "image_name" in labels_df.columns:
        labels_df = labels_df.drop_duplicates(subset="image_name", keep="first")
        print(f"After removing duplicate images in labels: {len(labels_df)} samples")
    label_columns = labels_df.columns.tolist()

    # Merge on image_name if available
    if "image_name" in features_df.columns and "image_name" in labels_df.columns:
        merged_df = pd.merge(features_df, labels_df, on="image_name", how="inner")
        features_df = merged_df[feature_columns]
        labels_df = merged_df[label_columns]
        print(f"After merging: {len(features_df)} samples")

    classifier = PersonalityTraitClassifier()
    results = classifier.train_and_evaluate_all(
        features_df, labels_df, test_size=test_size, random_state=random_state
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    classifier.save_models(str(out_dir / model_file))
    classifier.generate_report(results, str(out_dir / "classification_report.txt"))
    if generate_plots:
        classifier.plot_confusion_matrices(results, str(out_dir))

    print("\nClassification pipeline completed.")


def analyze_single_image_array(
    gray_image: np.ndarray,
    filename: str = "uploaded.png",
    use_classifiers: bool = True,
    models_path: str = "classification_results/personality_classifiers.pkl",
) -> dict:
    """
    Analyze a single handwriting image from a grayscale numpy array.

    Returns a dictionary with:
      - image_name
      - rule_based_traits: Big Five scores (0-1)
      - svm_categories (optional): predicted categories 0/1/2 if models exist
    """
    if gray_image is None or gray_image.size == 0:
        raise ValueError("Empty image data provided")

    # 1. Preprocess image (can pass image array directly)
    preprocessor = HandwritingPreprocessor()
    preprocessed = preprocessor.preprocess_image(gray_image)

    # 2. Extract features
    extractor = HandwritingFeatureExtractor()
    features = extractor.extract_all_features(preprocessed, gray_image)
    features["image_name"] = filename
    features_df = pd.DataFrame([features])

    # 3. Rule-based personality mapping
    mapper = PersonalityTraitMapper()
    rule_traits_df = mapper.predict_traits(features_df, use_classifier=False)
    rule_traits = rule_traits_df.iloc[0].to_dict()

    result: dict[str, object] = {
        "image_name": filename,
        "rule_based_traits": {k: float(v) for k, v in rule_traits.items()},
    }

    # 4. Optional ML-based classification (if trained models exist)
    if use_classifiers and Path(models_path).exists():
        classifier = PersonalityTraitClassifier()
        classifier.load_models(models_path)
        ml_preds_df = classifier.predict_traits(features_df, classifier_type="svm")
        ml_preds = ml_preds_df.iloc[0].to_dict()
        result["svm_categories"] = {k: int(v) for k, v in ml_preds.items()}
    else:
        result["svm_categories"] = None

    return result


# -------------------------------------------------------------
# Flask backend
# -------------------------------------------------------------

app = Flask(__name__)


@app.route("/", methods=["GET"])
def serve_frontend() -> object:
    """Serve the main HTML frontend."""
    # Assumes index.html is in the same folder as this script
    return send_from_directory(".", "index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze() -> object:
    """
    REST endpoint to analyze a single uploaded handwriting image.

    Expects multipart/form-data with a 'file' field.
    Returns JSON with rule-based trait scores and optional SVM categories.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        file_bytes = np.frombuffer(file.read(), np.uint8)
        gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return jsonify({"error": "Unable to decode image"}), 400

        result = analyze_single_image_array(
            gray_image=gray,
            filename=file.filename,
            use_classifiers=True,
            models_path="classification_results/personality_classifiers.pkl",
        )
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Start Flask development server
    # Access the app via http://127.0.0.1:5000/ in your browser
    app.run(host="0.0.0.0", port=5000, debug=True)


