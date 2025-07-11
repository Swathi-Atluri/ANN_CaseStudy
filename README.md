# ANN Case Study: Fish Habitat Classification with DeepFish Dataset
This repo contains two Python scripts I wrote for our Artificial Neural Networks course case study, focusing on classifying fish habitats using the DeepFish dataset. The dataset has images of fish in different habitats, and we explored two approaches: a simple ANN and a VGG16-based model with transfer learning. Each script addresses the problem differently, and I’ve included visualisations to help clarify the models and their performance. 
## Dataset

The DeepFish dataset is organised into `train`, `val`, and `test` folders, with images across 9 habitat classes (e.g., `9894`, `9907`). Each folder contains subfolders for the classes, and images are in RGB format. The dataset has:

- **Train**: 3124 images
- **Validation**: 770 images
- **Test**: 3872 images  

**Note:**
The DeepFish dataset originally contains over 40,000 images. However, due to limited computational resources, we trained our models on a reduced subset of the dataset.

## Requirements

To run these scripts, you’ll need Python 3.x and the following libraries:

- **TensorFlow**: For deep learning models.
- **NetworkX, Matplotlib**: For visualizations.
- **NumPy, Pandas, Scikit-learn**: For data handling.

The scripts assume the dataset is in `/kaggle/input/deepfish/DeepFish/Classification/organized` (typical for Kaggle). Update the `base_dir` path if your dataset is elsewhere.

## Scripts Overview

### Script 1: Simple ANN (`DF_BaseModel.ipynb`)

**What it does**: This script builds a basic ANN to classify images into 9 habitat classes. Instead of using convolutional layers, it flattens the 128x128 RGB images into a 1D vector and passes them through dense layers. It’s a stripped-down approach to see how a non-convolutional model performs on image data.

**Key Features**:

- **Data Prep**: Uses `ImageDataGenerator` with augmentation (rotation, flip, zoom) to preprocess images and normalize pixel values.
- **Model**: A Sequential model with:
  - Flatten layer (49152 inputs).
  - Dense (128, ReLU, L2 regularization).
  - Dropout (0.3).
  - Dense (64, ReLU, L2 regularization).
  - Dropout (0.3).
  - Dense (9, softmax).
- **Training**: Trains for 10 epochs with Adam (lr=0.0001), categorical crossentropy, and early stopping (patience=10). Test accuracy: \~81.53%.
- **Prediction**: Classifies a sample image (though the output label `9894` suggests a possible class mapping issue).
- **Visualizations**:
  - Plots training/validation accuracy and loss.
  - A NetworkX diagram showing a simplified ANN structure (1 input, 3 hidden1, 3 hidden2, 1 output) with symbolic weights.

### Script 2: VGG16 Transfer Learning (`DF_VGG.ipynb`)

**What it does**: This script uses VGG16, pre-trained on ImageNet, for transfer learning. We freeze VGG16’s convolutional layers and add custom dense layers to classify the 9 habitats. It’s a more sophisticated approach, leveraging pre-trained features for better performance.

**Key Features**:

- **Data Prep**: Similar to Script 1, uses `ImageDataGenerator` with augmentation for training and normalization for val/test. Images are 128x128.
- **Model**: A Sequential model stacking:
  - VGG16 (frozen, no top layers).
  - GlobalAveragePooling2D.
  - Dense (128, ReLU, L2 regularization).
  - Dropout (0.5).
  - Dense (64, ReLU, L2 regularization).
  - Dropout (0.5).
  - Dense (9, softmax).
- **Training**: Trains for 10 epochs with Adam (lr=0.0001), categorical crossentropy, and early stopping. Test accuracy: \~87.22%.
- **Visualizations**:
  - Separate plots for training/validation accuracy and loss.
  - A NetworkX diagram of a simplified VGG16-inspired structure (3 input, 5 conv, 3 dense1, 2 dense2, 1 output).

## Lessons Learned

- **Model Selection Matters**: Implementing a basic ANN in Script 1 highlighted the limitations of using simple architectures for image classification, particularly due to the loss of spatial information when flattening images. In contrast, leveraging the pre-trained VGG16 model in Script 2 significantly improved performance, emphasizing the advantages of transfer learning in computer vision tasks.

- **Efficient Data Handling**: Keras’ `ImageDataGenerator` proved invaluable for preprocessing and augmenting image data, streamlining the pipeline and improving model generalization.

- **Importance of Visualization**: Plotting training and validation accuracy/loss curves, along with visualizing the model architecture, provided valuable insights into the model’s learning behavior and helped diagnose issues effectively.

- **Overcoming Challenges**: Navigating TensorFlow warnings, resolving unexpected label issues in Script 1, and adapting to the constraints of the Kaggle environment required careful debugging and persistence.

