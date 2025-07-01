# CRxK Dataset (Single-Shot Crime Event Detector)

A comprehensive deep learning framework for detecting and classifying criminal activities in surveillance footage using a two-stage hierarchical classification approach.

## 🎯 Overview

SSCED introduces a novel benchmark dataset consisting of crime event-based surveillance images with well-informed annotations. The system implements a two-stage classification pipeline that first detects whether criminal activity is occurring, then classifies the specific type of crime.

### Key Features

- **Two-stage hierarchical classification**: Binary crime detection followed by multi-class crime type classification
- **Comprehensive dataset**: 2,054,013 frames across 6 categories (5 crime types + normal scenes)
- **Multiple model architectures**: Custom CNN, ResNet-18/50, EfficientNet-B0, and Vision Transformer
- **Real-time processing capability**: Optimized for surveillance video analysis
- **Detailed evaluation metrics**: Precision, recall, F1-score with per-class analysis

## 📊 Dataset

The dataset consists of surveillance footage categorized into 6 classes:

| Category | Label | Description |
|----------|-------|-------------|
| Assault | 0 | Physical violence incidents |
| Burglary | 1 | Breaking and entering activities |
| Kidnapping | 2 | Abduction scenarios |
| Robbery | 3 | Theft with force or threat |
| Swoon | 4 | Fainting/collapse incidents |
| Normal | 5 | Regular surveillance footage |

**Dataset Statistics:**
- Total frames: 2,054,013
- Training subset: 51,000 frames (8,500 per category)
- Train/test split: 80/20
- Frame extraction: Every 3rd frame from source videos

## 🏗️ Architecture

### Two-Stage Classification Pipeline

```
Input Frame → Stage 1: Binary Classifier → Stage 2: Multi-class Classifier → Final Prediction
                    ↓                              ↓
               [Crime/Normal]              [Assault/Burglary/Kidnap/
                                          Robbery/Swoon]
```

### Supported Models

1. **Custom CNN**: Lightweight 3-layer convolutional network
2. **ResNet-18**: Residual network with 18 layers
3. **ResNet-50**: Deeper residual network with 50 layers
4. **EfficientNet-B0**: Efficient convolutional architecture
5. **Vision Transformer (ViT-B/16)**: Transformer-based vision model

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision opencv-python pandas numpy tqdm matplotlib
pip install moviepy imageio-ffmpeg beautifulsoup4 lxml
```

### Data Preparation

1. Place your dataset in the `./data` directory
2. Generate annotations from videos:

```bash
python preprocess/make_annotation.py --path ./data
```

3. Extract frames from videos:

```bash
python preprocess/frame_crop.py --path ./data --category [category_name]
```

### Training

#### Binary Classification (Crime/Normal Detection)

```bash
# Custom CNN
python model/bin_cnn.py

# ResNet-18
python model/bin_res18.py

# ResNet-50
python model/bin_res50.py

# EfficientNet-B0
python model/bin_eff.py

# Vision Transformer
python model/bin_vit.py
```

#### Multi-class Classification (Crime Type Classification)

```bash
# Custom CNN
python model/multi_cnn.py

# ResNet-18
python model/multi_res18.py

# ResNet-50
python model/multi_res50.py

# EfficientNet-B0
python model/multi_eff.py

# Vision Transformer
python model/multi_vit.py
```

### Two-Stage Evaluation

```bash
# Evaluate complete pipeline
python model/bin_multi_cnn.py     # Custom CNN pipeline
python model/bin_multi_res18.py   # ResNet-18 pipeline
python model/bin_multi_res50.py   # ResNet-50 pipeline
python model/bin_multi_eff.py     # EfficientNet pipeline
python model/bin_multi_vit.py     # ViT pipeline
```

## 📁 Project Structure

```
SSCED/
├── data/                          # Dataset directory
│   ├── frame_data/               # Extracted frames
│   │   ├── assault_frame/
│   │   ├── burglary_frame/
│   │   ├── kidnap_frame/
│   │   ├── robbery_frame/
│   │   ├── swoon_frame/
│   │   └── normal_frame/
│   └── videos/                   # Original video files
├── preprocess/                   # Data preprocessing utilities
│   ├── datautils.py             # Dataset classes and evaluation metrics
│   ├── make_annotation.py       # Annotation generation from videos
│   ├── frame_crop.py            # Frame extraction from videos
│   └── video_crop.py            # Video preprocessing and cropping
├── model/                       # Model training scripts
│   ├── bin_*.py                 # Binary classification training
│   ├── multi_*.py               # Multi-class classification training
│   └── bin_multi_*.py           # Two-stage pipeline evaluation
└── README.md
```

## 🔧 Configuration

### Model Parameters

- **Input size**: 256×256 (224×224 for ViT)
- **Batch size**: 32-128 (varies by model)
- **Learning rate**: 0.001
- **Optimizer**: Adam
- **Loss function**: CrossEntropyLoss

### Training Settings

- **Binary classification epochs**: 45
- **Multi-class epochs**: 30-100 (varies by model)
- **GPU support**: CUDA-enabled training
- **Data augmentation**: Resize transformation

## 📈 Results

The system demonstrates strong performance across different architectures:

- **Binary Classification**: Effectively distinguishes crime vs. normal scenes
- **Multi-class Classification**: Accurate classification of specific crime types
- **Two-stage Pipeline**: Combines both stages for comprehensive crime detection

Performance varies by model architecture, with traditional CNNs showing robust results and Vision Transformers presenting unique challenges for this surveillance domain.

## 🛠️ Custom Dataset Classes

The project includes specialized dataset classes in `preprocess/datautils.py` for different training scenarios:

- `CCTVDataset`: Complete dataset loader
- `CrimeDataset`: Crime-only samples (excludes normal scenes)
- `NormalDataset`: Binary classification (crime vs. normal)
- `AssaultDataset`, `BurglaryDataset`, etc.: Individual crime type datasets

## 📊 Evaluation Metrics

The system provides comprehensive evaluation through:

- **Accuracy**: Overall classification performance
- **Precision/Recall**: Per-class performance metrics
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification analysis
- **Macro-averaging**: Balanced performance across all classes

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- Dataset sourced from AI Hub surveillance collection
- Built with PyTorch and torchvision
- Inspired by advances in computer vision and surveillance technology
