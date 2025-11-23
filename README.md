# Handwriting Personality Analyzer

A web-based handwriting analysis tool that analyzes handwriting samples and provides Big Five personality trait scores.

## Features

- Upload handwriting images (PNG, JPG, JPEG, BMP, TIF, TIFF)
- Automatic image preprocessing and feature extraction
- Big Five personality trait analysis:
  - Openness
  - Conscientiousness
  - Extraversion
  - Agreeableness
  - Neuroticism
- Beautiful, modern web interface
- RESTful API backend

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
python app.py
```

The server will start on `http://localhost:8000`

### 3. Open the HTML Interface

Simply open `handwriting_analyzer.html` in your web browser. The interface will automatically connect to the backend server.

## Usage

1. Open `handwriting_analyzer.html` in your browser
2. Click the upload area or drag and drop a handwriting image
3. Click "Analyze Handwriting"
4. View your personality trait scores

## API Endpoints

- `GET /` - API information
- `POST /analyze` - Analyze a handwriting image (expects multipart/form-data with 'file' field)
- `GET /health` - Health check endpoint

## Response Format

```json
{
  "success": true,
  "personality": {
    "Openness": 75,
    "Conscientiousness": 60,
    "Extraversion": 45,
    "Agreeableness": 80,
    "Neuroticism": 30
  },
  "features": {
    "baseline_angle": 2.5,
    "letter_size": 25.0,
    "line_spacing": 40.0,
    "word_spacing": 15.0,
    "top_margin": 20.0,
    "pressure_score": 3.5,
    "slant_angle": 10.0
  }
}
```

## Technical Details

The analyzer extracts seven handwriting features:
1. Baseline angle
2. Letter size
3. Line spacing
4. Word spacing
5. Top margin
6. Pen pressure (stroke width)
7. Slant angle

These features are then mapped to Big Five personality scores using established graphological principles.

## Notes

- Scores range from 0-100
- 0-33: Low trait level
- 34-66: Moderate trait level
- 67-100: High trait level
- Results are based on handwriting characteristics and should be interpreted as general tendencies

