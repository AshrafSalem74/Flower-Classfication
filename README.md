# Flower Classification using Deep Neural Network (DNN)

## Project Overview
This project involves building a deep learning model using PyTorch to classify images of flowers into four categories: **daisy, roses, sunflowers, and tulips**. A fully connected Deep Neural Network (DNN) was implemented for multi-class classification. The model was evaluated using accuracy, F1-score, and confusion matrix.

## Dataset
- The dataset consists of class-labeled images of flowers.
- Images were resized to **224x224 pixels**.
- The dataset was split into training and test sets to evaluate model performance.

## Model Architecture
- The model consists of **5 hidden layers** with gradually decreasing nodes.
- **ReLU activation function** is applied to hidden layers.
- **Softmax activation function** is used in the output layer for probabilistic interpretation.
- The input images were flattened before being passed into the network.

## Loss Function & Optimizer
- **CrossEntropyLoss** was used as the loss function, as it is effective for multi-class classification.
- **Adam optimizer** was used for weight updates due to its adaptive learning rate and momentum capabilities.

## Hyperparameters
- **Learning rate**: 0.001
- **Batch size**: 64
- **Number of epochs**: 20
- **Optimizer**: Adam

## Training Process
- The training function performed forward and backward propagation for each epoch.
- The loss function was calculated for training and test sets after each epoch.
- The training loss was observed to decrease over epochs, showing effective learning.

## Model Evaluation
- The model achieved an **accuracy of 60%** and an **F1-score of 59%**.
- **Confusion matrix** analysis showed some misclassifications, especially among visually similar flower categories.
- The loss curves were plotted to analyze model performance and generalization.

## Error Analysis & Improvements
- The model sometimes struggled with distinguishing visually similar flowers, leading to misclassification.
- **Potential improvements**:
  - Use **Dropout regularization** to prevent overfitting.
  - **Tune hyperparameters** such as learning rate and batch size.
  - Implement **Convolutional Neural Networks (CNNs)** for better feature extraction.
  - Utilize a **larger dataset** to improve accuracy and generalization.

## How to Run the Code
1. Install required dependencies:
   ```bash
   pip install torch torchvision numpy matplotlib
   ```
2. Load and preprocess the dataset.
3. Train the model using the provided training function.
4. Evaluate performance using accuracy, F1-score, and confusion matrix.
5. Visualize misclassified images for further analysis.

## Conclusion
This project demonstrated the capability of Deep Neural Networks in image classification tasks. While the model achieved a moderate accuracy, there is potential for improvement by incorporating CNNs and fine-tuning hyperparameters. The insights gained from error analysis provide a roadmap for future enhancements.

