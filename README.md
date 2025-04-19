# ANN Case Study: Fish Habitat Classification with DeepFish Dataset
This repo contains three Python scripts I wrote for our Artificial Neural Networks course case study, focusing on classifying fish habitats using the DeepFish dataset. The dataset has images of fish in different habitats, and we explored three approaches: a simple ANN, a VGG16-based model with transfer learning, and a SOM+SVM combo. Each script tackles the problem differently, and I’ve included visualizations to make sense of the models and their performance. 
## Dataset

The DeepFish dataset is organized into `train`, `val`, and `test` folders, with images across 9 habitat classes (e.g., `9894`, `9907`). Each folder contains subfolders for the classes, and images are in RGB format. The dataset has:

- **Train**: 3124 images
- **Validation**: 770 images
- **Test**: 3872 images

For the third script, we interpreted the dataset as having `empty` and `valid` subfolders per class, treating it as a binary classification task (empty vs. valid). This might reflect a different organization or a specific task focus.\n
**Note:**
The DeepFish dataset originally contains over 40,000 images. However, due to limited computational resources, we trained our models on a reduced subset of the dataset.

## Requirements

To run these scripts, you’ll need Python 3.x and the following libraries:

- **TensorFlow**: For deep learning models (Scripts 1 and 2).
- **MiniSom**: For the SOM in Script 3.
- **OpenCV**: For image processing in Script 3.
- **NetworkX, Matplotlib**: For visualizations.
- **NumPy, Pandas, Scikit-learn**: For data handling and SVMs.

The scripts assume the dataset is in `/kaggle/input/deepfish/DeepFish/Classification/organized` (typical for Kaggle). Update the `base_dir` path if your dataset is elsewhere.

## Scripts Overview

### Script 1: Simple ANN (`DF_BaseModel.py`)

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

### Script 2: VGG16 Transfer Learning (`DF_VGG.py`)

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

### Script 3: SOM + SVM (`DF_SOM.py`)

**What it does**: This script takes a different angle, using a Self-Organizing Map (SOM) for unsupervised feature extraction, followed by SVMs for binary classification (empty vs. valid). It assumes each habitat class has `empty` and `valid` subfolders, which might be a different dataset setup.

**Key Features**:

- **Data Prep**: Loads images in grayscale, resizes to 64x64, and flattens to 4096D vectors. Labels are 0 (empty) or 1 (valid).
- **SOM**: A 20x20 SOM grid (400 neurons) trained for 2000 iterations on normalized features. Extracts:
  - BMU Coordinates (2D): SOM grid position for each sample.
  - Activation Responses (400D): SOM grid activation for each sample.
- **SVM**: Trains three RBF-kernel SVMs on:
  - BMU Coordinates (71.05% accuracy).
  - Activation Responses (86.31% accuracy).
  - Original Features (86.80% accuracy).
- **Visualizations**:
  - A hexagonal SOM grid showing sample clustering (empty vs. valid).
  - A NetworkX diagram of the pipeline (Input → SOM → BMU/Activation, Input → Original, SVMs → Output).

## Lessons Learned

- **Model Complexity**: Script 1’s simple ANN was easy to code but underperformed due to flattening images. Script 2’s VGG16 showed that leveraging pre-trained models is a huge win for image tasks. Script 3’s SOM+SVM was a creative middle ground, blending unsupervised and supervised learning.
- **Data Handling**: Keras’ `ImageDataGenerator` (Scripts 1 and 2) was a lifesaver for image preprocessing, but Script 3’s manual loading with OpenCV gave me more control (and headaches).
- **Visualizations**: Plotting accuracy/loss and drawing network diagrams helped me understand what was going on. The SOM grid in Script 3 was especially fun to interpret.
- **Challenges**: I hit some snags, like the weird class label in Script 1 and the dataset structure confusion in Script 3. Debugging TensorFlow warnings and managing Kaggle’s environment took patience.
