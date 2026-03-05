# Neural Style Transfer using VGG-19

## Overview

This is a **beginner-friendly machine learning project** designed to explore practical applications of deep neural networks. The project implements **Neural Style Transfer (NST)**, an interesting technique that combines the content of one image with the artistic style of another using a pre-trained **VGG-19** convolutional neural network.

## Project Purpose

This project was created as a learning exercise to:
- Understand how pre-trained models can be repurposed for creative applications
- Learn about feature extraction from deep neural networks
- Explore gradient-based optimization for content generation
- Practice implementing neural network concepts in TensorFlow/Keras
- Gain hands-on experience with practical ML applications in Google Colab

## What the Code Does

The neural style transfer pipeline works by:

1. **Loading Images**: Accepts a content image (e.g., a photograph) and a style image (e.g., a famous painting)
2. **Feature Extraction**: Uses VGG-19 to extract meaningful features from both images at different layers
3. **Loss Calculation**: Computes two types of losses:
   - **Content Loss**: Measures how well the structure of the generated image matches the content image
   - **Style Loss**: Measures how well the texture and color patterns of the generated image match the style image
4. **Optimization**: Iteratively updates the generated image to minimize both losses, creating an output that has the content of one image with the artistic style of another

## Key Components

### 1. **Image Preprocessing**
- `load_img()`: Loads images from file, resizes them while preserving aspect ratio (max 512px), and converts to normalized tensors
- `tensor_to_image()`: Converts optimized tensors back into viewable image format

### 2. **VGG-19 Feature Extraction**
- The pre-trained VGG-19 model (trained on ImageNet) is used without its classification layers
- Intermediate layers capture different types of visual information:
  - **Content Layer**: `block4_conv2` captures high-level structural features
  - **Style Layers**: `block1_conv1` through `block5_conv1` capture texture patterns at multiple scales

### 3. **Gram Matrix**
The Gram matrix is a mathematical construct that captures the correlations between feature maps:
- **Formula**: Computes the dot product of flattened feature maps
- **Purpose**: Provides a compact representation of style that is position-invariant
- Allows the model to capture "what the image looks like" rather than "where things are"

### 4. **StyleContentModel Class**
A custom Keras model that:
- Wraps VGG-19 to extract features from specified layers
- Applies Gram matrix transformation to style features
- Returns both raw content features and Gram matrix style representations
- Freezes VGG parameters (no backpropagation through the network itself)

### 5. **Loss Function**
Combines two weighted losses:
```
Total Loss = (Style Loss × weight) + (Content Loss × weight)
```
- **Style Loss**: Mean squared difference between Gram matrices of generated and style images
- **Content Loss**: Mean squared difference between intermediate feature maps of generated and content images
- Weights control the balance between preserving content vs. matching style

### 6. **Optimization Loop**
- Uses Adam optimizer to update the generated image directly
- Runs for multiple epochs (default: 5) with multiple steps per epoch (default: 100)
- Each step computes gradients with respect to the image pixels and updates them
- Clips pixel values to stay within valid range [0, 1] after each update

### 7. **Interactive UI**
- Built with `ipywidgets` for Google Colab
- Allows users to:
  - Upload multiple images at once
  - Select content and style images from dropdowns
  - Preview selected images
  - Run style transfer with one button click
  - View the stylized output immediately

## How to Use

### Requirements
- TensorFlow 2.x
- NumPy
- Matplotlib
- PIL (Pillow)
- ipywidgets
- Google Colab (optional, but recommended for free GPU access)

### Running the Code
1. Open the notebook in Google Colab (click the badge at the top)
2. Run all cells in order - libraries will be set up automatically
3. When prompted, upload your content and style images
4. Select images from the dropdown menus
5. Click "Run Style Transfer" button
6. Wait for processing (faster with GPU)
7. View the stylized result

### Parameters You Can Adjust
- **epochs**: Number of complete passes through optimization (default: 5)
- **steps_per_epoch**: Number of gradient updates per epoch (default: 50-100)
- **learning_rate**: Adam optimizer learning rate (default: 0.02)
- **style_weight**: How much to emphasize style vs content (default: 1e2)
- **content_weight**: How much to emphasize content structure (default: 1e4)

## Technical Details

### Layer Selection
The code uses carefully chosen VGG-19 layers:
- **Content extraction** from deeper layers (`block4_conv2`) because deeper layers capture semantically meaningful content
- **Style extraction** from multiple layers (shallow to deep) to capture textures at multiple scales

### Why VGG-19?
- Pre-trained on ImageNet with millions of images
- Simple architecture that learns good feature representations
- Intermediate layers have interpretable outputs
- Well-established for style transfer tasks

### Computational Approach
- **No training**: VGG-19 weights are frozen; we only optimize the image pixels
- **Gradient-based optimization**: Similar to how neural networks train, but we update the image instead of weights
- **GPU acceleration**: Beneficial due to many tensor operations

## Expected Results
The algorithm typically produces:
- Higher quality results after more epochs (diminishing returns after ~10 epochs)
- Better style transfer when style and content images have different characteristics
- Faster processing with GPU acceleration (10-100x faster than CPU)

## Limitations
- Style transfer works best with artistic style images (paintings, artworks)
- Very different content and style images may produce unpredictable results
- High-resolution images take significantly longer to process
- The method may sometimes introduce artifacts or oversaturated colors

## Learning Outcomes

This project demonstrates:
- How deep learning models learn hierarchical features
- Practical applications of CNNs beyond classification
- Gradient-based optimization techniques
- Working with pre-trained models in TensorFlow/Keras
- Building interactive ML applications in Jupyter/Colab

## References
- Gatys et al., "A Neural Algorithm of Artistic Style" (2015)
- VGG-19 paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (2014)
- TensorFlow/Keras documentation for style transfer

## Author Notes

This project served as an excellent introduction to practical machine learning applications. It bridges the gap between understanding neural network theory and implementing real-world creative applications, making it an ideal starting point for anyone interested in deep learning.