#  CNN Projects — Deep Learning with TensorFlow/Keras

A collection of Convolutional Neural Network projects built from scratch and using transfer learning — covering image classification, production deployment, and model explainability.

---

##  Projects

### 1. 🌿 Plant Disease Detection (EfficientNetB0)
> Transfer learning model detecting 15 plant diseases from leaf images

- **Accuracy:** 93.09% validation accuracy
- **Model:** EfficientNetB0 (ImageNet pretrained, frozen)
- **Classes:** 15 (Tomato, Potato, Pepper Bell diseases)
- **Trainable Params:** 19,215 out of 4,000,000
- **Explainability:** Grad-CAM implemented
- **Deployed:** Flask REST API → Docker → Render (live)
- **Key decisions:** Confidence thresholding (0.60), two-phase training strategy

🔗 **Live API:** [https://plant-disease-api-l7mi.onrender.com](https://plant-disease-api-l7mi.onrender.com)
🔗 **Deployment Repo:** [plant-disease-api](https://github.com/Kuldip-Lakhtariya/plant-disease-api)

---

### 2.  CIFAR-10 Image Classifier (Custom CNN)
> Built a custom CNN from scratch achieving 80.6% validation accuracy on 10-class image classification

- **Accuracy:** 80.6% validation accuracy
- **Architecture:** Conv→BN→Pool ×3, Dropout(0.25), Dense(128), Softmax
- **Dataset:** CIFAR-10 (60,000 images, 10 classes)
- **Key learnings:**
  - BatchNorm placement and effect
  - Data augmentation strategy
  - GAP vs Flatten trade-offs
  - Output shape calculation
  - Transfer learning limitations on low-resolution images (32×32)
  - Confusion matrix analysis

---

##  Tech Stack

`TensorFlow` · `Keras` · `Flask` · `Docker` · `Gunicorn` · `Render` · `NumPy` · `Matplotlib` · `Grad-CAM`

---

##  Skills Demonstrated

| Skill | Project |
|-------|---------|
| Custom CNN architecture design | CIFAR-10 |
| Transfer learning & fine-tuning | Plant Disease |
| Production API development | Plant Disease |
| Docker containerization | Plant Disease |
| Cloud deployment | Plant Disease |
| Model explainability (Grad-CAM) | Plant Disease |
| Confidence thresholding | Plant Disease |
| BatchNorm, Dropout, Augmentation | Both |
| Confusion matrix analysis | Both |

---

## 👤 Author

**Kuldip Lakhtariya** — B.Tech Student | ML & DL Engineer

🔗 [GitHub](https://github.com/Kuldip-Lakhtariya) · [Live Plant Disease App](https://plant-disease-api-l7mi.onrender.com)
