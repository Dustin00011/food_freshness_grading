# 🍎 FreshCheck AI – Food Freshness Classification System

A deep learning project that classifies fruit and vegetable images as **Fresh**, **Semi-Fresh** or **Rotten**. Two models are trained and compared: a **custom CNN built from scratch** and a **MobileNetV2 transfer-learning model** (frozen backbone, feature extraction). Both are served in a Streamlit web app that shows their predictions side by side.

*Deep Learning PBL (CSAI3027P)*

---

## 1. Problem statement

Judging freshness by eye is subjective and inconsistent, and it contributes to food waste when good produce is thrown away or spoiled produce is missed. This project builds an image-based classifier that predicts the freshness stage of a fruit or vegetable from a single photo.

## 2. Dataset

AgriFreshNET Freshness and Shelf-Life Image Dataset (processed version).

| Item | Value |
|---|---|
| Total images | 14,160 (no corrupted files) |
| Image size | 512 × 512 |
| Produce types | 8 – banana, bittermelon, cucumber, eggplant, orange, papaya, pineapple, tomato |
| Classes | 3 – Fresh, Semi-Fresh, Rotten (perfectly balanced, 4,720 each) |
| Split | Stratified 70 / 15 / 15 → 9,912 train / 2,124 validation / 2,124 test (seed 42) |

Class folders are grouped by days since harvest (for example `Fresh Banana(1-4)`), so the freshness label reflects **age**, not only visible appearance.

## 3. Models

| | Custom CNN | MobileNetV2 |
|---|---|---|
| Idea | Trained from scratch | ImageNet backbone frozen, new classification head trained |
| Input | 128 × 128 | 224 × 224 |
| Architecture | 4 blocks × (2 × Conv-BatchNorm-ReLU) + MaxPool, filters 32→64→128→256, spatial dropout, GlobalAveragePooling, Dense(256), Dropout(0.4), Softmax(3) | MobileNetV2 → GlobalAveragePooling → Dropout(0.3) → Dense(256) → Dropout(0.3) → Softmax(3) |
| Parameters | 1.24 M (all trainable) | 2.59 M (0.33 M trainable) |

**Training setup (both):** Adam (lr 1e-3), sparse categorical cross-entropy, `ReduceLROnPlateau`, `EarlyStopping` with best-weights restore, batch size 32, fixed seed. Augmentation (flip, rotation, zoom, translation, mild brightness/contrast) is applied in the `tf.data` pipeline. Hue and saturation are not changed because colour is the main freshness cue.

## 4. Results (test set, 2,124 images)

| Metric | Custom CNN | MobileNetV2 |
|---|---|---|
| **Accuracy** | **82.16 %** | **93.50 %** |
| Macro F1 | 0.8188 | 0.9348 |
| F1 – Fresh | 0.8283 | 0.9290 |
| F1 – Semi-Fresh | 0.7585 | 0.9088 |
| F1 – Rotten | 0.8696 | 0.9665 |

Optimization history: the first MobileNetV2 run reached about 88 % and the first custom CNN runs 65–74 %. Fixing the data pipeline (cache/shuffle order), adding augmentation, a richer head, a learning-rate schedule and longer training raised these to the figures above.

**Error analysis**
- *Semi-Fresh* is the hardest class for both models, since it sits between the other two.
- Rotten is the easiest class (MobileNetV2 recall ≈ 97.7 %).
- Lowest MobileNetV2 accuracy by produce: eggplant (87.6 %), tomato (89.7 %), pineapple (90.3 %).
- The custom CNN is weakest on eggplant (63.6 %) and papaya (68.3 %).
- Many mistakes are boundary cases where the label follows days since harvest but the image still looks fresh.

Plots (confusion matrices, training curves, accuracy by produce, misclassified examples) are in `figures/`.

## 5. Project structure

```
├── app.py                                   # Streamlit app
├── models/
│   ├── best_custom_cnn_fixed.keras          # custom CNN
│   └── best_mobilenetv2.keras               # MobileNetV2
├── 01_dataset_analysis.ipynb                # dataset exploration
├── food_freshness_training_improved.ipynb   # training + evaluation
├── figures/                                 # result plots, results.json
├── requirements.txt
└── README.md
```

## 6. Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a JPG/PNG of a fruit or vegetable and click **Analyse Freshness**. The app shows each model's predicted class, per-class confidence bars, and whether the two models agree.

Use the same TensorFlow/Keras version that was used for training (TensorFlow 2.21) to avoid model-loading problems.

## 7. Reproduce the training

1. Open `food_freshness_training_improved.ipynb` on Kaggle with the dataset attached and a GPU enabled.
2. Run all cells. It trains both models, reloads the saved files from disk, evaluates them on the test split, and writes the models and figures.
3. Copy the `.keras` files into `models/`.

**Preprocessing must match training.** Both models expect pixel values scaled to [-1, 1] (`x / 127.5 - 1`, the same maths as MobileNetV2 `preprocess_input`) and RGB input. Class order is `["Fresh", "Semi-Fresh", "Rotten"]`.

## 8. Limitations

- Only the 8 produce types in the dataset are covered. Any other image is still forced into one of the three classes.
- Labels are based on days since harvest, so Semi-Fresh is inherently ambiguous.
- Some near-identical images appear in the dataset, which suggests the test accuracy may be optimistic. The models have **not yet been evaluated on independent real-world photos**.
- Images come from controlled backgrounds. Lighting and background changes in real use may reduce accuracy.

## 9. Ethics and responsible use

- FreshCheck AI is a **visual aid, not a food-safety tool**. Do not use it to decide whether food is safe to eat.
- Predictions can be wrong, especially for Semi-Fresh items and for produce outside the 8 trained types. Confidence scores are shown so users can judge reliability.
- The dataset covers a limited set of produce, varieties and growing regions, so results may not generalise everywhere.
- Intended benefit: reduce food waste by supporting consistent freshness checks.

## 10. Future work

- Fine-tune the top MobileNetV2 layers for higher accuracy.
- Build a small test set of real phone photos and report accuracy on it.
- Add more produce types and an "unknown / not produce" option or a confidence threshold.
- Add Grad-CAM visualisations to show what the model looks at.
- Improve the app interface.

## 11. Tech stack

Python · TensorFlow/Keras · scikit-learn · pandas · NumPy · Matplotlib · Streamlit · Pillow
