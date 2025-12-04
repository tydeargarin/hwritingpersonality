"""
Handwriting Image Preprocessing Pipeline
Based on: "A Framework for Determining the Big Five Personality Traits 
Using Machine Learning Classification through Graphology"

Preprocessing stages:
1. Noise removal (Bilateral filtering)
2. Thresholding (Binary conversion)
3. Segmentation (Text region extraction)
4. Cropping (Remove top and bottom margins)
5. Normalization (Resize to target width without side padding)
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import argparse


class HandwritingPreprocessor:
    """
    Preprocessing class for handwriting images following the research paper methodology.
    """
    
    def __init__(self, 
                 bilateral_d=9, 
                 bilateral_sigma_color=75, 
                 bilateral_sigma_space=75,
                 threshold_method='adaptive',
                 normalization_size=(800, 600),
                 top_crop=650,
                 bottom_crop=650):
        """
        Initialize the preprocessor with parameters.
        
        Args:
            bilateral_d: Diameter of pixel neighborhood for bilateral filter
            bilateral_sigma_color: Filter sigma in color space
            bilateral_sigma_space: Filter sigma in coordinate space
            threshold_method: 'adaptive' or 'otsu' for thresholding
            normalization_size: Target size (width, height) for normalization
            top_crop: Number of pixels to crop from top (default: 650)
            bottom_crop: Number of pixels to crop from bottom (default: 650)
        """
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.threshold_method = threshold_method
        self.normalization_size = normalization_size
        self.top_crop = top_crop
        self.bottom_crop = bottom_crop
    
    def remove_noise(self, image):
        """
        Step 1: Noise removal using bilateral filtering.
        Bilateral filter preserves edges while removing noise.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Denoised image
        """
        # Apply bilateral filter for noise removal
        denoised = cv2.bilateralFilter(
            image, 
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
        return denoised
    
    def apply_thresholding(self, image):
        """
        Step 2: Thresholding to convert grayscale to binary image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Binary image (0 or 255)
        """
        if self.threshold_method == 'adaptive':
            # Adaptive thresholding - better for varying lighting conditions
            binary = cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
        elif self.threshold_method == 'otsu':
            # Otsu's thresholding - automatically determines optimal threshold
            _, binary = cv2.threshold(
                image,
                0,
                255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            # Simple thresholding
            _, binary = cv2.threshold(
                image,
                127,
                255,
                cv2.THRESH_BINARY_INV
            )
        
        return binary
    
    def segment_text_region(self, binary_image):
        """
        Step 3: Segmentation to extract text region and remove background.
        Uses morphological operations and contour detection.
        
        Args:
            binary_image: Binary image
            
        Returns:
            Segmented image with text region extracted
        """
        # Create a copy for processing
        segmented = binary_image.copy()
        
        # Morphological operations to clean up the image
        # Remove small noise
        kernel = np.ones((3, 3), np.uint8)
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_CLOSE, kernel)
        segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)
        
        # Find contours to identify text regions
        contours, _ = cv2.findContours(segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for text regions
        mask = np.zeros_like(segmented)
        
        # Filter contours by area (remove very small noise)
        min_area = 50
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Apply mask to get segmented text
        segmented = cv2.bitwise_and(segmented, mask)
        
        return segmented
    
    def crop_image(self, image, top_crop=650, bottom_crop=650):
        """
        Crop image by removing specified pixels from top and bottom.
        
        Args:
            image: Binary image
            top_crop: Number of pixels to remove from top
            bottom_crop: Number of pixels to remove from bottom
            
        Returns:
            Cropped image
        """
        height, width = image.shape[:2]
        
        # Ensure we don't crop more than available
        top_crop = min(top_crop, height)
        bottom_crop = min(bottom_crop, height - top_crop)
        
        # Crop: remove top and bottom
        cropped = image[top_crop:height-bottom_crop, :]
        
        return cropped
    
    def normalize_image(self, image):
        """
        Step 4: Normalization - resize to target width without side padding.
        Maintains aspect ratio and fits to target width exactly.
        
        Args:
            image: Binary image
            
        Returns:
            Normalized image with target width (no side white bars)
        """
        # Get current dimensions
        height, width = image.shape[:2]
        target_width, target_height = self.normalization_size
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Resize to fit target width exactly (maintaining aspect ratio)
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # If resized height is less than target height, pad bottom only
        # If resized height is more than target height, crop from bottom
        if new_height < target_height:
            # Pad bottom with white
            normalized = np.ones((target_height, target_width), dtype=np.uint8) * 255
            normalized[:new_height, :] = resized
        elif new_height > target_height:
            # Crop from bottom to fit target height
            normalized = resized[:target_height, :]
        else:
            # Exact fit
            normalized = resized
        
        return normalized
    
    def preprocess_image(self, image_path, top_crop=650, bottom_crop=650):
        """
        Complete preprocessing pipeline for a single image.
        
        Args:
            image_path: Path to input image
            top_crop: Number of pixels to crop from top (default: 650)
            bottom_crop: Number of pixels to crop from bottom (default: 650)
            
        Returns:
            Preprocessed image
        """
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = image_path.copy()
        
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Step 1: Noise removal
        denoised = self.remove_noise(image)
        
        # Step 2: Thresholding
        binary = self.apply_thresholding(denoised)
        
        # Step 3: Segmentation
        segmented = self.segment_text_region(binary)
        
        # Step 3.5: Crop top and bottom
        cropped = self.crop_image(segmented, top_crop=top_crop, bottom_crop=bottom_crop)
        
        # Step 4: Normalization (no side white bars)
        normalized = self.normalize_image(cropped)
        
        return normalized
    
    def preprocess_dataset(self, input_dir, output_dir, file_extensions=('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
        """
        Preprocess entire dataset of handwriting images.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save preprocessed images
            file_extensions: Tuple of file extensions to process
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process each image
        successful = 0
        failed = 0
        
        for image_path in tqdm(image_files, desc="Preprocessing images"):
            try:
                # Preprocess image
                preprocessed = self.preprocess_image(
                    str(image_path), 
                    top_crop=self.top_crop, 
                    bottom_crop=self.bottom_crop
                )
                
                # Save preprocessed image
                output_path = Path(output_dir) / image_path.name
                cv2.imwrite(str(output_path), preprocessed)
                successful += 1
                
            except Exception as e:
                print(f"\nError processing {image_path}: {str(e)}")
                failed += 1
        
        print(f"\nPreprocessing complete!")
        print(f"Successfully processed: {successful} images")
        print(f"Failed: {failed} images")


def main():
    """
    Main function to run preprocessing from command line.
    """
    parser = argparse.ArgumentParser(
        description='Preprocess handwriting images using techniques from the research paper'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='handwriting',
        help='Input directory containing handwriting images (default: handwriting)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='handwriting_preprocessed',
        help='Output directory for preprocessed images (default: handwriting_preprocessed)'
    )
    parser.add_argument(
        '--threshold_method',
        type=str,
        choices=['adaptive', 'otsu', 'simple'],
        default='adaptive',
        help='Thresholding method: adaptive, otsu, or simple (default: adaptive)'
    )
    parser.add_argument(
        '--normalization_width',
        type=int,
        default=800,
        help='Target width for normalization (default: 800)'
    )
    parser.add_argument(
        '--normalization_height',
        type=int,
        default=600,
        help='Target height for normalization (default: 600)'
    )
    parser.add_argument(
        '--bilateral_d',
        type=int,
        default=9,
        help='Bilateral filter diameter (default: 9)'
    )
    parser.add_argument(
        '--bilateral_sigma_color',
        type=int,
        default=75,
        help='Bilateral filter sigma color (default: 75)'
    )
    parser.add_argument(
        '--bilateral_sigma_space',
        type=int,
        default=75,
        help='Bilateral filter sigma space (default: 75)'
    )
    parser.add_argument(
        '--top_crop',
        type=int,
        default=650,
        help='Number of pixels to crop from top (default: 650)'
    )
    parser.add_argument(
        '--bottom_crop',
        type=int,
        default=650,
        help='Number of pixels to crop from bottom (default: 650)'
    )
    
    args = parser.parse_args()
    
    # Create preprocessor with specified parameters
    preprocessor = HandwritingPreprocessor(
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        threshold_method=args.threshold_method,
        normalization_size=(args.normalization_width, args.normalization_height),
        top_crop=args.top_crop,
        bottom_crop=args.bottom_crop
    )
    
    # Preprocess dataset
    preprocessor.preprocess_dataset(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()

