# Plant Disease Detection

EfficientNetB0 transfer learning achieving **93.09% validation accuracy** 
across 15 plant disease classes. Deployed as a Flask web API.

## Demo
Upload a leaf image → get disease prediction + confidence score

![Plant Disease Detector UI](demo.png)

## Architecture
Input (224×224×3)
→ EfficientNetB0 (frozen, ImageNet weights)
→ GlobalAveragePooling2D
→ Dropout(0.2)
→ Dense(15, softmax)
Trainable params: 19,215 / 4,049,571 total
## Training Strategy
| Phase | Layers Unfrozen | LR | Epochs | Val Accuracy |
|-------|----------------|-----|--------|--------------|
| 1 | Head only | 1e-3 | 10 | 93.09% |
| 2 | Skipped — plateau reached | — | — | — |

## Results
| Class | Accuracy |
|-------|----------|
| Pepper healthy | 100% |
| Potato Early Blight | 99% |
| Tomato Early Blight | 60% ← hardest class |
| Overall | 93.09% |

**Why Early Blight is hardest:** Early Blight, Late Blight, and Target Spot 
all produce brown circular spots on tomato leaves. Visual features overlap 
even for human experts.

## Production Features
- Confidence thresholding — predictions below 60% rejected with warning
- Grad-CAM explainability — heatmap shows which leaf regions drove prediction
- Flask REST API — `/predict` endpoint accepts image, returns JSON
- HTML frontend — farmer-friendly upload interface

## API Usage
```bash
POST /predict
Content-Type: multipart/form-data
Body: image file

Response:
{
  "predicted_class": "Tomato_Early_blight",
  "confidence": 0.9412,
  "warning": null
}
```

## Stack
Python · TensorFlow/Keras · Flask · HTML/CSS · Jupyter

## Dataset
PlantVillage — 15 classes, ~17,000 images  
Source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
