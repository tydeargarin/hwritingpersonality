"""
Example usage of the handwriting preprocessing pipeline.
"""

from preprocess_handwriting import HandwritingPreprocessor
import cv2
import matplotlib.pyplot as plt

# Example 1: Preprocess a single image
def example_single_image():
    """Example of preprocessing a single image."""
    # Initialize preprocessor
    preprocessor = HandwritingPreprocessor(
        threshold_method='adaptive',
        normalization_size=(800, 600)
    )
    
    # Preprocess single image
    image_path = 'handwriting/0052-1.tif'  # Replace with your image path
    preprocessed = preprocessor.preprocess_image(image_path)
    
    # Display results
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(preprocessed, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison saved to 'preprocessing_comparison.png'")
    
    # Save preprocessed image
    cv2.imwrite('preprocessed_example.png', preprocessed)
    print("Preprocessed image saved to 'preprocessed_example.png'")


# Example 2: Preprocess entire dataset
def example_dataset():
    """Example of preprocessing entire dataset."""
    # Initialize preprocessor
    preprocessor = HandwritingPreprocessor(
        threshold_method='adaptive',
        normalization_size=(800, 600)
    )
    
    # Preprocess entire dataset
    preprocessor.preprocess_dataset(
        input_dir='handwriting',
        output_dir='handwriting_preprocessed'
    )


if __name__ == '__main__':
    print("Handwriting Preprocessing Examples")
    print("=" * 40)
    
    # Uncomment the example you want to run:
    
    # Example 1: Preprocess single image
    # example_single_image()
    
    # Example 2: Preprocess entire dataset
    example_dataset()

