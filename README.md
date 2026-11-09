# Indian License Plate Reader

Detect and read Indian license plates from images and videos using OpenCV + Tesseract OCR. No external model download required.

## Features

- 🇮🇳 **Tuned for Indian plates** — validates `DL01CQ1234`, `MH12AB3456`, `KA05EF9012`, `TN10AA1234`, etc.
- 📷 **Image and video support** — process a single photo or an entire video file frame-by-frame
- 🔲 **Contour-based detection** — blackhat morphology + Canny edge fallback (no Haar cascade XML needed)
- 🧹 **IND-strip removal** — crops off the left-side security band found on many Indian plates
- 🔤 **Fuzzy OCR recovery** — handles common OCR confusions (`O→0`, `I→1`, etc.) across multiple Tesseract modes
- 🟢 **Annotated output** — green boxes + recognized text drawn directly on the result image/video

## How it works

Input image/video → Plate detection (contours) → Crop → Strip IND band → Preprocess (CLAHE, threshold) → OCR (Tesseract) → Validate against Indian plate regex → Annotate output

Indian plate format validated a-posteriori:
^A-Z{2}0-9{1,2}A-Z{1,2}0-9{1,4}$
Examples: `KA05EF9012`, `DL01CQ1234`, `GJ01ZZ9999`, `KL01AB1234`

## Installation

```bash
# Clone the repo
git clone https://github.com/20041212S/Veh-Plate-Reader.git
cd Veh-Plate-Reader

# Install Python dependencies
pip install opencv-python pytesseract

# Install Tesseract OCR engine (system package)
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
Usage
# Process an image
python3 plate_reader.py --input photo.jpg --output result.jpg

# Process a video
python3 plate_reader.py --input clip.mp4 --output result.mp4
Example output
Plate at (234, 236, 525, 335) -> text: 'KA05EF9012'
Web deployment
This is a Python/Flask backend. It is not suitable for Vercel serverless functions because:
- OpenCV and Tesseract are large system dependencies
- Serverless functions have cold-start and timeout limits unsuitable for frame-by-frame video processing
To run as a website with phone camera access:
1. Deploy the Flask backend (app.py) on Railway, Render, Fly.io, or a VPS
2. Serve the HTML/JS frontend separately (Vercel static site works for this)
3. Point the frontend's fetch() calls to your backend URL
Limitations
- Detection works best on frontal/near-frontal plates
- Angled, tilted, or heavily obscured plates may be missed
- No plate tracking across video frames (each frame processed independently)
- For production accuracy on fast-moving traffic, swap the contour detector for a YOLOv8 model fine-tuned on Indian plates
License
MIT
