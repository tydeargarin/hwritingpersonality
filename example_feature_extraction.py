"""
Example usage of the handwriting feature extraction pipeline.
"""

from extract_features import HandwritingFeatureExtractor
from preprocess_handwriting import HandwritingPreprocessor
import cv2
import pandas as pd
import matplotlib.pyplot as plt

# Example 1: Extract features from a single image
def example_single_image():
    """Example of extracting features from a single image."""
    # Initialize preprocessor and feature extractor
    preprocessor = HandwritingPreprocessor()
    extractor = HandwritingFeatureExtractor()
    
    # Load and preprocess image
    image_path = 'handwriting/0052-1.tif'  # Replace with your image path
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    preprocessed = preprocessor.preprocess_image(image_path)
    
    # Extract features
    features = extractor.extract_all_features(preprocessed, original)
    
    # Display features
    print("\nExtracted Features:")
    print("=" * 50)
    for feature_name, value in features.items():
        print(f"{feature_name:25s}: {value:.4f}")
    
    return features


# Example 2: Extract features from entire dataset
def example_dataset():
    """Example of extracting features from entire dataset."""
    extractor = HandwritingFeatureExtractor()
    
    # Extract features using preprocessed images
    df = extractor.extract_features_from_dataset(
        input_dir='handwriting',
        output_csv='handwriting_features.csv',
        use_preprocessed=True,
        preprocessed_dir='handwriting_preprocessed'
    )
    
    # Display summary statistics
    print("\nFeature Summary Statistics:")
    print("=" * 50)
    print(df.describe())
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df


# Example 3: Visualize feature distributions
def example_visualize_features():
    """Example of visualizing feature distributions."""
    # Extract features
    extractor = HandwritingFeatureExtractor()
    df = extractor.extract_features_from_dataset(
        input_dir='handwriting',
        output_csv='handwriting_features.csv',
        use_preprocessed=True,
        preprocessed_dir='handwriting_preprocessed'
    )
    
    # Select numeric columns for visualization
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    numeric_cols = [c for c in numeric_cols if c != 'image_name']
    
    # Create subplots
    n_features = len(numeric_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(numeric_cols):
        if idx < len(axes):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[idx].set_title(f'{col}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
    print("Feature distributions saved to 'feature_distributions.png'")


if __name__ == '__main__':
    print("Handwriting Feature Extraction Examples")
    print("=" * 50)
    
    # Uncomment the example you want to run:
    
    # Example 1: Extract features from single image
    # example_single_image()
    
    # Example 2: Extract features from entire dataset
    example_dataset()
    
    # Example 3: Visualize feature distributions
    # example_visualize_features()

