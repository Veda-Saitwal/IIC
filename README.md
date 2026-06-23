# Interactive Image Colorizer

An AI-powered web application that automatically colorizes grayscale images and allows users to interactively recolor detected objects while preserving realistic shading and textures.

## Features

- Automatic grayscale image colorization using Deep Learning
- Object detection and segmentation
- Interactive object selection
- Custom color application through color picker
- Preserves image shading and texture during recoloring
- User-friendly web interface
- Real-time image processing

## Technologies Used

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask

### Computer Vision & AI
- OpenCV
- NumPy
- Deep Learning Colorization Model

## Project Structure

```text
Interactive-Image-Colorizer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── colorization_deploy_v2.prototxt
│   ├── pts_in_hull.npy
│   └── colorization_release_v2.caffemodel
│
├── static/
│   ├── uploads/
│   └── output/
│
└── templates/
    └── index.html
```

## Model File

The trained model file is larger than GitHub's file size limit and is therefore hosted separately.

### Download Model

Download the model from:

https://drive.google.com/drive/folders/1nSunaWGLzoevZWdLnJKC4V3LB9jbwOIE?usp=sharing

After downloading, place it in:

model/
├── colorization_release_v2.caffemodel
├── colorization_deploy_v2.prototxt
└── pts_in_hull.npy

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Interactive-Image-Colorizer.git
cd Interactive-Image-Colorizer
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model

Download the model from the Google Drive link and place it inside the `model` folder.

### 5. Run Application

```bash
python app.py
```

### 6. Open Browser

```text
http://127.0.0.1:5000
```

## Working

1. Upload a grayscale image.
2. The AI model automatically colorizes the image.
3. Objects are detected and segmented.
4. Select an object from the image.
5. Choose a custom color.
6. Apply the color while preserving natural shading.
7. Download or view the final output.

## Applications

- Historical photograph restoration
- Digital art enhancement
- Image editing and recoloring
- Photography enhancement
- Educational demonstrations of Computer Vision

## Future Enhancements

- Support for multiple object recoloring
- Advanced segmentation models
- User authentication
- Batch image processing
- Cloud deployment
- Mobile-friendly interface

## Requirements

- Python 3.10+
- Flask
- OpenCV
- NumPy

Install all required packages using:

```bash
pip install -r requirements.txt
```

## Author

**Veda Saitwal**

Third Year Computer Engineering Student  
PES Modern College of Engineering, Pune

## License

This project is developed for educational and academic purposes.
