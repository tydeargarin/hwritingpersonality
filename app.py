from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from typing import Tuple
import io
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_data: np.ndarray) -> np.ndarray:
    """Preprocess image using the same pipeline as the original analyzer"""
    # Convert to grayscale if needed
    if len(image_data.shape) == 3:
        gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_data.copy()
    
    # Apply bilateral filter for noise removal
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Otsu binarization
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (foreground should be white for contour detection)
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)
    
    # Auto-crop to content
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find bounding box of all contours
        x_coords = []
        y_coords = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_coords.extend([x, x + w])
            y_coords.extend([y, y + h])
        
        if x_coords and y_coords:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            binary = binary[y_min:y_max, x_min:x_max]
    
    # Normalize orientation (simplified version)
    # Use dilation to connect text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Estimate skew angle
    coords = cv2.findNonZero(dilated)
    if coords is not None and len(coords) > 5:
        rot_rect = cv2.minAreaRect(coords)
        angle = rot_rect[2]
        if angle < -45.0:
            angle = 90.0 + angle
        
        # Rotate to deskew
        h, w = binary.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
        binary = cv2.warpAffine(binary, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    
    # Final binarization
    _, final_binary = cv2.threshold(binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return final_binary

def extract_features(binary_image: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """Extract the seven handwriting features from preprocessed image"""
    # Ensure foreground is white
    if binary_image.mean() > 127:
        binary_image = cv2.bitwise_not(binary_image)
    
    # Detect text lines
    projection = binary_image.sum(axis=1)
    threshold = 0.05 * float(projection.max())
    lines = []
    in_line = False
    start = 0
    for y, val in enumerate(projection):
        if val > threshold and not in_line:
            in_line = True
            start = y
        elif val <= threshold and in_line:
            in_line = False
            lines.append((start, y))
    if in_line:
        lines.append((start, len(projection) - 1))
    
    # Merge small gaps
    merged_lines = []
    for line in lines:
        if not merged_lines:
            merged_lines.append(line)
            continue
        prev_start, prev_end = merged_lines[-1]
        if line[0] - prev_end <= 2:
            merged_lines[-1] = (prev_start, line[1])
        else:
            merged_lines.append(line)
    
    # 1. Baseline angle
    edges = cv2.Canny(binary_image, 50, 150, apertureSize=3)
    hough_lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=100)
    baseline_angle = 0.0
    if hough_lines is not None:
        angles = []
        for line in hough_lines[:200]:
            for rho, theta in line:
                deg = float(theta * 180.0 / np.pi)
                if deg >= 180.0:
                    deg -= 180.0
                if deg >= 90.0:
                    deg -= 180.0
                if abs(deg) <= 30.0:
                    angles.append(deg)
        if angles:
            baseline_angle = float(np.median(angles))
    
    # 2. Letter size
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    heights = []
    img_h, img_w = binary_image.shape[:2]
    min_area = max(10, int(0.0002 * img_h * img_w))
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area >= min_area and h > 1:
            heights.append(h)
    letter_size = float(np.median(heights)) if heights else 0.0
    
    # 3. Line spacing
    line_spacing = 0.0
    if len(merged_lines) >= 2:
        centers = [int((a + b) / 2) for a, b in merged_lines]
        spacings = np.diff(centers)
        line_spacing = float(np.median(spacings))
    
    # 4. Word spacing
    word_spacings = []
    for y0, y1 in merged_lines:
        line_img = binary_image[y0:y1, :]
        if line_img.size == 0:
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(line_img, connectivity=8)
        components = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            if area >= 5 and w > 1 and h > 1:
                components.append((x, w))
        if len(components) >= 2:
            components.sort(key=lambda b: b[0])
            for i in range(1, len(components)):
                gap = components[i][0] - (components[i-1][0] + components[i-1][1])
                if gap > 0:
                    word_spacings.append(gap)
    word_spacing = float(np.median(word_spacings)) if word_spacings else 0.0
    
    # 5. Top margin
    top_margin = float(merged_lines[0][0]) if merged_lines else 0.0
    
    # 6. Pen pressure (stroke width via distance transform)
    inv = (binary_image > 0).astype(np.uint8)
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    pressure_score = 2.0 * float(dist[inv > 0].mean()) if dist.size > 0 and inv.sum() > 0 else 0.0
    
    # 7. Slant angle
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    angles = []
    for contour in contours:
        if len(contour) >= 10:
            pts = contour.reshape(-1, 2).astype(np.float32)
            mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)
            if eigenvectors is not None and eigenvectors.shape[0] > 0:
                vx, vy = float(eigenvectors[0, 0]), float(eigenvectors[0, 1])
                angle_rad = np.arctan2(vy, vx)
                angle_deg = float(angle_rad * 180.0 / np.pi)
                rel = angle_deg - 90.0
                while rel <= -90.0:
                    rel += 180.0
                while rel > 90.0:
                    rel -= 180.0
                if abs(rel) <= 60.0:
                    angles.append(rel)
    slant_angle = float(np.median(angles)) if angles else 0.0
    
    return baseline_angle, letter_size, line_spacing, word_spacing, top_margin, pressure_score, slant_angle

def map_to_personality(features: Tuple[float, float, float, float, float, float, float]) -> dict:
    """Map features to Big Five personality scores"""
    baseline_angle, letter_size, line_spacing, word_spacing, top_margin, pressure_score, slant_angle = features
    
    def clamp01(x):
        return max(0.0, min(1.0, x))
    
    def score_descending_baseline(angle):
        return clamp01((-angle) / 10.0)
    
    def score_ascending_baseline(angle):
        return clamp01(angle / 10.0)
    
    def score_straight_baseline(angle):
        return clamp01(1.0 - (abs(angle) / 6.0))
    
    def score_slant_inclined(angle):
        return clamp01(max(0.0, angle) / 30.0)
    
    def score_slant_reclined(angle):
        return clamp01(max(0.0, -angle) / 30.0)
    
    def score_slant_moderate_magnitude(angle):
        mag = abs(angle)
        if mag <= 5.0 or mag >= 35.0:
            return 0.0
        if mag <= 20.0:
            return (mag - 5.0) / 15.0
        return (35.0 - mag) / 15.0
    
    def score_slant_extreme(angle):
        return clamp01(abs(angle) / 30.0)
    
    def score_large_size(size):
        return clamp01(size / 50.0)
    
    def score_small_size(size):
        return clamp01(1.0 - (size / 50.0))
    
    def score_large_spacing(spacing):
        return clamp01(spacing / 100.0)
    
    def score_small_spacing(spacing):
        return clamp01(1.0 - (spacing / 100.0))
    
    def score_normal_spacing(spacing):
        return clamp01(1.0 - abs(spacing - 40.0) / 40.0)
    
    def score_large_margin(margin):
        return clamp01(margin / 100.0)
    
    def score_large_pressure(pressure):
        return clamp01(pressure / 10.0)
    
    def score_small_pressure(pressure):
        return clamp01(1.0 - (pressure / 10.0))
    
    # Calculate Big Five scores
    neuroticism = 0.6 * score_descending_baseline(baseline_angle) + \
                  0.4 * (score_slant_inclined(slant_angle) * score_slant_moderate_magnitude(slant_angle))
    
    openness = 0.4 * score_small_spacing(line_spacing) + \
               0.2 * score_normal_spacing(word_spacing) + \
               0.4 * (score_slant_inclined(slant_angle) * score_slant_moderate_magnitude(slant_angle))
    
    extraversion = 0.25 * score_ascending_baseline(baseline_angle) + \
                   0.25 * score_large_size(letter_size) + \
                   0.25 * score_large_pressure(pressure_score) + \
                   0.25 * score_slant_extreme(slant_angle)
    
    agreeableness = 0.4 * score_large_margin(top_margin) + \
                    0.3 * score_small_pressure(pressure_score) + \
                    0.3 * (score_slant_reclined(slant_angle) * score_slant_moderate_magnitude(slant_angle))
    
    conscientiousness = 0.35 * score_straight_baseline(baseline_angle) + \
                        0.2 * score_small_size(letter_size) + \
                        0.25 * score_large_pressure(pressure_score) + \
                        0.2 * score_slant_reclined(slant_angle)
    
    return {
        'Neuroticism': int(round(100.0 * clamp01(neuroticism))),
        'Openness': int(round(100.0 * clamp01(openness))),
        'Extraversion': int(round(100.0 * clamp01(extraversion))),
        'Agreeableness': int(round(100.0 * clamp01(agreeableness))),
        'Conscientiousness': int(round(100.0 * clamp01(conscientiousness)))
    }

@app.route('/')
def index():
    """Serve the HTML interface"""
    return send_from_directory('.', 'handwriting_analyzer.html')

@app.route('/analyze', methods=['POST'])
def analyze_handwriting():
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: PNG, JPG, JPEG, BMP, TIF, TIFF'}), 400
        
        # Read image data
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Could not decode image. Please ensure the file is a valid image.'}), 400
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Extract features
        features = extract_features(processed_image)
        
        # Map to personality scores
        personality_scores = map_to_personality(features)
        
        # Return results
        return jsonify({
            'success': True,
            'personality': personality_scores,
            'features': {
                'baseline_angle': float(features[0]),
                'letter_size': float(features[1]),
                'line_spacing': float(features[2]),
                'word_spacing': float(features[3]),
                'top_margin': float(features[4]),
                'pressure_score': float(features[5]),
                'slant_angle': float(features[6])
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Processing failed: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Handwriting Analyzer server...")
    print("Server will be available at http://localhost:8000")
    print("Open http://localhost:8000 in your browser to use the interface")
    app.run(host='0.0.0.0', port=8000, debug=True)

