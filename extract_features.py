"""
Feature Extraction for Handwriting Analysis
Based on: "A Framework for Determining the Big Five Personality Traits 
Using Machine Learning Classification through Graphology"

Extracts seven features:
1. Baseline
2. Top Margin
3. Line Spacing
4. Word Spacing
5. Letter Size
6. Slant
7. Pen Pressure
"""

import cv2
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse


class HandwritingFeatureExtractor:
    """
    Feature extraction class for handwriting images following the research paper methodology.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_baseline(self, binary_image):
        """
        Extract baseline feature - the angle/alignment of the writing baseline.
        Based on research paper methodology using horizontal projection and line detection.
        Baseline indicates emotional stability and consistency.
        
        Args:
            binary_image: Binary image (0 = background, 255 = text)
            
        Returns:
            baseline_angle: Angle of baseline in degrees
            baseline_stability: Standard deviation of baseline (lower = more stable)
        """
        # Invert for processing (text = 1, background = 0)
        img = (binary_image == 0).astype(np.uint8)
        
        # Project text pixels onto horizontal axis
        horizontal_projection = np.sum(img, axis=1)
        
        # Smooth the projection to reduce noise (as per paper methodology)
        kernel_size = 5
        smoothed = np.convolve(horizontal_projection, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Find peaks in horizontal projection (text lines)
        # Use adaptive threshold based on max projection
        peak_threshold = np.max(smoothed) * 0.3
        min_distance = max(20, img.shape[0] // 20)  # Adaptive minimum distance
        
        peaks, properties = find_peaks(smoothed, height=peak_threshold, distance=min_distance)
        
        if len(peaks) == 0:
            return 0.0, 0.0
        
        # Calculate baseline for each line (bottom of text region)
        baseline_points = []
        baseline_x_coords = []
        
        for idx, peak in enumerate(peaks):
            # Find the region around this peak
            start_row = max(0, peak - 10)
            end_row = min(img.shape[0], peak + 50)
            
            # Find the bottom-most row with significant text (baseline)
            region_projection = horizontal_projection[start_row:end_row]
            if len(region_projection) == 0:
                continue
                
            # Baseline is where text density drops significantly
            max_in_region = np.max(region_projection)
            threshold = max_in_region * 0.5
            
            baseline_row = peak
            for r in range(peak, end_row):
                if r < len(horizontal_projection) and horizontal_projection[r] < threshold:
                    baseline_row = r
                    break
            
            # Use center x-coordinate of the line for baseline point
            line_region = img[start_row:end_row, :]
            if np.sum(line_region) > 0:
                # Find center x-coordinate of text in this line
                vertical_proj = np.sum(line_region, axis=0)
                x_coords = np.where(vertical_proj > np.max(vertical_proj) * 0.1)[0]
                if len(x_coords) > 0:
                    center_x = (x_coords[0] + x_coords[-1]) // 2
                    baseline_points.append(baseline_row)
                    baseline_x_coords.append(center_x)
        
        if len(baseline_points) < 2:
            return 0.0, 0.0
        
        # Calculate baseline angle using linear regression on x-y coordinates
        x_coords = np.array(baseline_x_coords)
        y_coords = np.array(baseline_points)
        
        # Fit line to baseline points
        if len(x_coords) > 1 and np.std(x_coords) > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_coords, y_coords)
            baseline_angle = np.degrees(np.arctan(slope))
        else:
            baseline_angle = 0.0
        
        # Calculate stability (standard deviation of residuals from fitted line)
        if len(x_coords) > 1 and np.std(x_coords) > 0:
            predicted = slope * x_coords + intercept
            residuals = y_coords - predicted
            baseline_stability = np.std(residuals)
        else:
            baseline_stability = np.std(y_coords) if len(y_coords) > 1 else 0.0
        
        return baseline_angle, baseline_stability
    
    def extract_top_margin(self, binary_image):
        """
        Extract top margin feature - the space between the top of the page and first line.
        Based on research paper: measures distance from top edge to first text baseline.
        Top margin indicates social behavior and respect for boundaries.
        
        Args:
            binary_image: Binary image (0 = background, 255 = text)
            
        Returns:
            top_margin: Distance from top of image to first text line (normalized)
        """
        # Invert for processing
        img = (binary_image == 0).astype(np.uint8)
        
        # Project text pixels onto horizontal axis
        horizontal_projection = np.sum(img, axis=1)
        
        # Smooth the projection to reduce noise
        kernel_size = 5
        smoothed = np.convolve(horizontal_projection, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Find first row with significant text (top margin ends where text begins)
        # Use adaptive threshold based on maximum projection
        threshold = np.max(smoothed) * 0.1
        top_margin = 0
        
        # Find the first peak or significant text region
        for i, proj in enumerate(smoothed):
            if proj > threshold:
                # Found start of text, this is the top margin
                top_margin = i
                break
        
        # If no text found, check if there's any text at all
        if top_margin == 0 and np.max(smoothed) > 0:
            # Find first non-zero projection
            for i, proj in enumerate(smoothed):
                if proj > 0:
                    top_margin = i
                    break
        
        # Normalize by image height
        normalized_margin = top_margin / binary_image.shape[0] if binary_image.shape[0] > 0 else 0
        
        return normalized_margin
    
    def extract_line_spacing(self, binary_image):
        """
        Extract line spacing feature - the vertical distance between text lines.
        Based on research paper: measures spacing between baselines of consecutive lines.
        Line spacing indicates organization and thinking patterns.
        
        Args:
            binary_image: Binary image (0 = background, 255 = text)
            
        Returns:
            avg_line_spacing: Average spacing between lines (normalized)
            line_spacing_variance: Variance in line spacing
        """
        # Invert for processing
        img = (binary_image == 0).astype(np.uint8)
        
        # Project text pixels onto horizontal axis
        horizontal_projection = np.sum(img, axis=1)
        
        # Smooth the projection
        kernel_size = 5
        smoothed = np.convolve(horizontal_projection, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Find peaks (text lines) with adaptive parameters
        peak_threshold = np.max(smoothed) * 0.3
        min_distance = max(20, img.shape[0] // 20)
        peaks, properties = find_peaks(smoothed, height=peak_threshold, distance=min_distance)
        
        if len(peaks) < 2:
            return 0.0, 0.0
        
        # Calculate baseline for each line (as in baseline extraction)
        baseline_rows = []
        for peak in peaks:
            start_row = max(0, peak - 10)
            end_row = min(img.shape[0], peak + 50)
            region_projection = horizontal_projection[start_row:end_row]
            
            if len(region_projection) == 0:
                continue
                
            max_in_region = np.max(region_projection)
            threshold = max_in_region * 0.5
            
            baseline_row = peak
            for r in range(peak, end_row):
                if r < len(horizontal_projection) and horizontal_projection[r] < threshold:
                    baseline_row = r
                    break
            baseline_rows.append(baseline_row)
        
        if len(baseline_rows) < 2:
            return 0.0, 0.0
        
        # Calculate spacing between consecutive baselines
        line_spacings = []
        for i in range(len(baseline_rows) - 1):
            spacing = baseline_rows[i + 1] - baseline_rows[i]
            if spacing > 0:  # Ensure positive spacing
                line_spacings.append(spacing)
        
        if len(line_spacings) == 0:
            return 0.0, 0.0
        
        # Normalize by image height
        avg_line_spacing = np.mean(line_spacings) / binary_image.shape[0] if binary_image.shape[0] > 0 else 0
        line_spacing_variance = np.var(line_spacings) / (binary_image.shape[0] ** 2) if binary_image.shape[0] > 0 else 0
        
        return avg_line_spacing, line_spacing_variance
    
    def extract_word_spacing(self, binary_image):
        """
        Extract word spacing feature - the horizontal distance between words.
        Based on research paper: identifies word boundaries using vertical projection.
        Word spacing indicates social interaction and communication style.
        
        Args:
            binary_image: Binary image (0 = background, 255 = text)
            
        Returns:
            avg_word_spacing: Average spacing between words (normalized)
            word_spacing_variance: Variance in word spacing
        """
        # Invert for processing
        img = (binary_image == 0).astype(np.uint8)
        
        # Project text pixels onto vertical axis
        vertical_projection = np.sum(img, axis=0)
        
        # Smooth the projection to reduce noise
        kernel_size = 3
        smoothed_projection = np.convolve(vertical_projection, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Adaptive threshold: gaps are regions with low projection
        max_projection = np.max(smoothed_projection)
        threshold = max_projection * 0.1  # 10% of max indicates gap
        
        # Find gaps (spaces between words)
        gaps = []
        in_gap = False
        gap_start = 0
        min_gap_size = max(3, img.shape[1] // 200)  # Adaptive minimum gap size
        
        for i, proj in enumerate(smoothed_projection):
            if proj < threshold and not in_gap:
                # Start of a gap
                in_gap = True
                gap_start = i
            elif proj >= threshold and in_gap:
                # End of a gap
                in_gap = False
                gap_length = i - gap_start
                if gap_length >= min_gap_size:  # Filter out very small gaps (likely within words)
                    gaps.append(gap_length)
        
        # Handle gap at the end
        if in_gap:
            gap_length = len(smoothed_projection) - gap_start
            if gap_length >= min_gap_size:
                gaps.append(gap_length)
        
        if len(gaps) == 0:
            return 0.0, 0.0
        
        # Normalize by image width
        avg_word_spacing = np.mean(gaps) / binary_image.shape[1] if binary_image.shape[1] > 0 else 0
        word_spacing_variance = np.var(gaps) / (binary_image.shape[1] ** 2) if binary_image.shape[1] > 0 else 0
        
        return avg_word_spacing, word_spacing_variance
    
    def extract_letter_size(self, binary_image):
        """
        Extract letter size feature - the average size of letters.
        Based on research paper: segments individual characters and calculates dimensions.
        Letter size indicates self-esteem and confidence.
        
        Args:
            binary_image: Binary image (0 = background, 255 = text)
            
        Returns:
            avg_letter_height: Average letter height (normalized)
            avg_letter_width: Average letter width (normalized)
            letter_size_variance: Variance in letter sizes
        """
        # Invert for processing
        img = (binary_image == 0).astype(np.uint8)
        
        # Apply morphological operations to separate connected characters
        # Use opening to break thin connections
        kernel = np.ones((2, 2), np.uint8)
        img_processed = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find connected components (letters/characters)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_processed, connectivity=8)
        
        if num_labels < 2:  # At least one component (background + text)
            return 0.0, 0.0, 0.0
        
        # Adaptive filtering based on image size
        img_area = binary_image.shape[0] * binary_image.shape[1]
        min_area = max(10, img_area // 10000)  # Minimum area for a letter
        max_area = min(5000, img_area // 10)   # Maximum area (likely merged)
        min_height = max(5, binary_image.shape[0] // 100)
        max_height = min(200, binary_image.shape[0] // 3)
        min_width = max(3, binary_image.shape[1] // 200)
        max_width = min(200, binary_image.shape[1] // 4)
        
        # Filter out very small components (noise) and very large ones (likely merged letters)
        heights = []
        widths = []
        areas = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter reasonable letter sizes with adaptive thresholds
            if (min_area < area < max_area and 
                min_height < height < max_height and 
                min_width < width < max_width):
                heights.append(height)
                widths.append(width)
                areas.append(area)
        
        if len(heights) == 0:
            return 0.0, 0.0, 0.0
        
        # Normalize by image dimensions
        avg_letter_height = np.mean(heights) / binary_image.shape[0] if binary_image.shape[0] > 0 else 0
        avg_letter_width = np.mean(widths) / binary_image.shape[1] if binary_image.shape[1] > 0 else 0
        
        # Calculate variance in letter sizes (using area as size metric)
        letter_size_variance = np.var(areas) / (img_area ** 2) if img_area > 0 else 0
        
        return avg_letter_height, avg_letter_width, letter_size_variance
    
    def extract_slant(self, binary_image):
        """
        Extract slant feature - the angle at which letters are written.
        Based on research paper: analyzes stroke orientation using line detection.
        Slant indicates emotional expression and social orientation.
        
        Args:
            binary_image: Binary image (0 = background, 255 = text)
            
        Returns:
            avg_slant_angle: Average slant angle in degrees
            slant_consistency: Consistency of slant (lower = more consistent)
        """
        # Invert for processing
        img = (binary_image == 0).astype(np.uint8)
        
        # Use Hough Line Transform to detect line segments
        # Adaptive parameters based on image size
        min_line_length = max(20, min(img.shape[0], img.shape[1]) // 20)
        max_line_gap = max(5, min(img.shape[0], img.shape[1]) // 50)
        threshold = max(30, min(img.shape[0], img.shape[1]) // 20)
        
        lines = cv2.HoughLinesP(
            img, 
            rho=1, 
            theta=np.pi/180, 
            threshold=threshold,
            minLineLength=min_line_length, 
            maxLineGap=max_line_gap
        )
        
        if lines is None or len(lines) == 0:
            return 0.0, 0.0
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate angle
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) > 1:  # Avoid division by zero
                angle = np.degrees(np.arctan2(dy, dx))
                
                # Focus on near-vertical strokes (character strokes)
                # Normalize angle to -90 to 90 range
                if angle > 90:
                    angle = angle - 180
                elif angle < -90:
                    angle = angle + 180
                
                # Consider angles between -60 and 60 degrees (character strokes)
                if -60 < angle < 60:
                    angles.append(angle)
        
        if len(angles) == 0:
            return 0.0, 0.0
        
        # Calculate average slant (positive = right slant, negative = left slant)
        avg_slant_angle = np.mean(angles)
        
        # Calculate consistency (standard deviation of angles)
        slant_consistency = np.std(angles)
        
        return avg_slant_angle, slant_consistency
    
    def extract_pen_pressure(self, binary_image, original_grayscale=None):
        """
        Extract pen pressure feature - the intensity/darkness of the writing.
        Based on research paper: analyzes stroke thickness and intensity.
        Pen pressure indicates energy level and emotional intensity.
        
        Note: For binary images, we estimate pressure from stroke width using distance transform.
        If original grayscale is provided, we use intensity values (darker = higher pressure).
        
        Args:
            binary_image: Binary image (0 = background, 255 = text)
            original_grayscale: Optional original grayscale image for better pressure estimation
            
        Returns:
            avg_pressure: Average pen pressure (normalized 0-1)
            pressure_variance: Variance in pen pressure
        """
        if original_grayscale is not None:
            # Check if dimensions match
            binary_shape = binary_image.shape
            original_shape = original_grayscale.shape
            
            # If dimensions don't match, resize original to match binary
            if binary_shape != original_shape:
                original_grayscale = cv2.resize(
                    original_grayscale, 
                    (binary_shape[1], binary_shape[0]), 
                    interpolation=cv2.INTER_AREA
                )
            
            # Use grayscale intensity as pressure indicator
            # Mask with text regions (text = 0 in binary_image)
            text_mask = (binary_image == 0).astype(np.uint8)
            text_pixels = original_grayscale[text_mask > 0]
            
            if len(text_pixels) == 0:
                return 0.0, 0.0
            
            # Invert: darker pixels (lower values) = higher pressure
            # Normalize to 0-1 range (higher = more pressure)
            normalized_pressure = 1.0 - (text_pixels.astype(float) / 255.0)
            avg_pressure = np.mean(normalized_pressure)
            pressure_variance = np.var(normalized_pressure)
        else:
            # Estimate pressure from stroke width in binary image
            # Thicker strokes = higher pressure (as per research paper methodology)
            img = (binary_image == 0).astype(np.uint8)
            
            # Use distance transform to estimate stroke width
            # DIST_L2 provides Euclidean distance from edge to center
            dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
            
            # Get stroke widths (distance from edge to center)
            # Only consider pixels that are part of strokes
            stroke_widths = dist_transform[dist_transform > 0]
            
            if len(stroke_widths) == 0:
                return 0.0, 0.0
            
            # Normalize stroke widths to 0-1 range
            max_width = np.max(stroke_widths)
            if max_width > 0:
                normalized_widths = stroke_widths / max_width
                avg_pressure = np.mean(normalized_widths)
                pressure_variance = np.var(normalized_widths)
            else:
                avg_pressure = 0.0
                pressure_variance = 0.0
        
        return avg_pressure, pressure_variance
    
    def extract_all_features(self, binary_image, original_grayscale=None):
        """
        Extract all seven features from a handwriting image.
        
        Args:
            binary_image: Preprocessed binary image
            original_grayscale: Optional original grayscale image for pen pressure
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # 1. Baseline
        baseline_angle, baseline_stability = self.extract_baseline(binary_image)
        features['baseline_angle'] = baseline_angle
        features['baseline_stability'] = baseline_stability
        
        # 2. Top Margin
        top_margin = self.extract_top_margin(binary_image)
        features['top_margin'] = top_margin
        
        # 3. Line Spacing
        avg_line_spacing, line_spacing_variance = self.extract_line_spacing(binary_image)
        features['avg_line_spacing'] = avg_line_spacing
        features['line_spacing_variance'] = line_spacing_variance
        
        # 4. Word Spacing
        avg_word_spacing, word_spacing_variance = self.extract_word_spacing(binary_image)
        features['avg_word_spacing'] = avg_word_spacing
        features['word_spacing_variance'] = word_spacing_variance
        
        # 5. Letter Size
        avg_letter_height, avg_letter_width, letter_size_variance = self.extract_letter_size(binary_image)
        features['avg_letter_height'] = avg_letter_height
        features['avg_letter_width'] = avg_letter_width
        features['letter_size_variance'] = letter_size_variance
        
        # 6. Slant
        avg_slant_angle, slant_consistency = self.extract_slant(binary_image)
        features['avg_slant_angle'] = avg_slant_angle
        features['slant_consistency'] = slant_consistency
        
        # 7. Pen Pressure
        avg_pressure, pressure_variance = self.extract_pen_pressure(binary_image, original_grayscale)
        features['avg_pressure'] = avg_pressure
        features['pressure_variance'] = pressure_variance
        
        return features
    
    def extract_features_from_dataset(self, input_dir, output_csv=None, use_preprocessed=True, preprocessed_dir=None):
        """
        Extract features from all images in a dataset.
        
        Args:
            input_dir: Directory containing input images
            output_csv: Path to save features CSV file
            use_preprocessed: Whether to use preprocessed images
            preprocessed_dir: Directory containing preprocessed images (if use_preprocessed=True)
            
        Returns:
            DataFrame containing features for all images
        """
        from preprocess_handwriting import HandwritingPreprocessor
        
        # Initialize preprocessor if needed
        preprocessor = HandwritingPreprocessor() if not use_preprocessed else None
        
        # Get all image files
        image_files = []
        for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg']:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return pd.DataFrame()
        
        print(f"Found {len(image_files)} images to process")
        
        # Extract features from each image
        all_features = []
        successful = 0
        failed = 0
        
        for image_path in tqdm(image_files, desc="Extracting features"):
            try:
                # Load image
                if use_preprocessed and preprocessed_dir:
                    img_path = Path(preprocessed_dir) / image_path.name
                    binary_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    original_img = None
                else:
                    original_img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                    # Preprocess if needed
                    binary_img = preprocessor.preprocess_image(str(image_path))
                
                if binary_img is None:
                    raise ValueError(f"Could not read image: {image_path}")
                
                # Extract features
                features = self.extract_all_features(binary_img, original_img)
                features['image_name'] = image_path.name
                all_features.append(features)
                successful += 1
                
            except Exception as e:
                print(f"\nError processing {image_path}: {str(e)}")
                failed += 1
        
        # Create DataFrame
        df = pd.DataFrame(all_features)
        
        # Reorder columns to have image_name first
        if 'image_name' in df.columns:
            cols = ['image_name'] + [c for c in df.columns if c != 'image_name']
            df = df[cols]
        
        # Save to CSV if specified
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\nFeatures saved to {output_csv}")
        
        print(f"\nFeature extraction complete!")
        print(f"Successfully processed: {successful} images")
        print(f"Failed: {failed} images")
        
        return df


def main():
    """
    Main function to run feature extraction from command line.
    """
    parser = argparse.ArgumentParser(
        description='Extract handwriting features using techniques from the research paper'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='handwriting',
        help='Input directory containing handwriting images (default: handwriting)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='handwriting_features.csv',
        help='Output CSV file for features (default: handwriting_features.csv)'
    )
    parser.add_argument(
        '--use_preprocessed',
        action='store_true',
        help='Use preprocessed images instead of raw images'
    )
    parser.add_argument(
        '--preprocessed_dir',
        type=str,
        default='handwriting_preprocessed',
        help='Directory containing preprocessed images (default: handwriting_preprocessed)'
    )
    
    args = parser.parse_args()
    
    # Create feature extractor
    extractor = HandwritingFeatureExtractor()
    
    # Extract features from dataset
    extractor.extract_features_from_dataset(
        input_dir=args.input_dir,
        output_csv=args.output_csv,
        use_preprocessed=args.use_preprocessed,
        preprocessed_dir=args.preprocessed_dir
    )


if __name__ == '__main__':
    main()

