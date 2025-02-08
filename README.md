
### 1.2. Data Splitting

To ensure robust training and evaluation, the dataset was partitioned into training and testing sets. The distribution of images across these sets for each class is summarized in Table 1. A train-test split ratio of approximately 70:30 was utilized, ensuring a sufficient number of samples for training while retaining a substantial test set for unbiased performance assessment. Due to the size of the dataset, parallel processing with 4 workers was utilized to accelerate the data splitting process.

**Table 1: Image Distribution Across Training and Testing Sets**

| Class  | Training Set Size | Testing Set Size |
| :------- | :---------------: | :--------------: |
| ain    | 1479  | 635   |
| al    | 940   | 403   |
| aleff   | 1170  | 502   |
| bb     | 1253  | 538   |
| dal    | 1143  | 491   |
| dha    | 1206  | 517   |
| dhad   | 1169  | 501   |
| fa     | 1368  | 587   |
| gaaf   | 1193  | 512   |
| ghain  | 1383  | 594   |
| ha     | 1114  | 478   |
| haa    | 1068  | 458   |
| jeem   | 1086  | 466   |
| kaaf   | 1241  | 533   |
| khaa   | 1124  | 483   |
| la     | 1222  | 524   |
| laam   | 1282  | 550   |
| meem   | 1235  | 530   |
| nun    | 1273  | 546   |
| ra     | 1161  | 498   |
| saad   | 1326  | 569   |
| seen   | 1146  | 492   |
| sheen  | 1054  | 453   |
| ta     | 1271  | 545   |
| taa    | 1286  | 552   |
| thaa   | 1236  | 530   |
| thal   | 1107  | 475   |
| toot   | 1253  | 538   |
| waw    | 959   | 412   |
| ya     | 1205  | 517   |
| yaa    | 905   | 388   |
| zay    | 961   | 413   |

### 1.3. Preprocessing

The images were preprocessed using the following torchvision transforms:

*   Resizing to 224x224 pixels.
*   Conversion to PyTorch tensors.
*   Normalization using the ImageNet statistics (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).

## 2. Model Architecture

The core of the system is a Convolutional Neural Network (CNN) named ArSLNet, implemented using PyTorch. The architecture is described below:

*   **Feature Extraction (self.features):**
    *   Four sequential blocks, each containing:
        *   `Conv2d`: Convolutional layer with a 3x3 kernel, padding of 1. The channel sizes progress as follows: 3 -> 32, 32 -> 64, 64 -> 128, and 128 -> 256.
        *   `ReLU`: Rectified Linear Unit activation for non-linearity.
        *   `MaxPool2d`: Max pooling layer with a 2x2 kernel and stride of 2 to reduce spatial dimensions.
*   **Classification (self.classifier):**
    *   `Dropout (0.5)`: Dropout layer to prevent overfitting.
    *   `Linear`: A fully connected layer that maps the flattened output from the convolutional layers to 512 units.
    *   `ReLU`: ReLU activation for non-linearity.
    *   `Dropout (0.5)`: Another dropout layer.
    *   `Linear`: A final linear layer that maps 512 units to the number of classes (32).

The table below summarizes the layer architecture of the model.

| Layer (type)       | Output Shape          |  Param #    |
| :----------------- | :-------------------- | :----------: |
| Conv2d-1           | [1, 32, 224, 224]    | 896 |
| ReLU-2             | [1, 32, 224, 224]    | 0 |
| MaxPool2d-3        | [1, 32, 112, 112]    | 0 |
| Conv2d-4           | [1, 64, 112, 112]    | 18,496  |
| ReLU-5             | [1, 64, 112, 112]    | 0 |
| MaxPool2d-6        | [1, 64, 56, 56]     | 0 |
| Conv2d-7           | [1, 128, 56, 56]    | 73,856 |
| ReLU-8             | [1, 128, 56, 56]    | 0 |
| MaxPool2d-9        | [1, 128, 28, 28]    | 0 |
| Conv2d-10          | [1, 256, 28, 28]    | 295,168 |
| ReLU-11            | [1, 256, 28, 28]    | 0 |
| MaxPool2d-12       | [1, 256, 14, 14]    | 0 |
| Flatten-13         | [1, 50176]          | 0 |
| Dropout-14         | [1, 50176]          | 0 |
| Linear-15          | [1, 512]         |  25,690,112|
| ReLU-16            | [1, 512]         | 0 |
| Dropout-17         | [1, 512]         | 0 |
| Linear-18          | [1, 32]         |  16,416|
| **Total Params** | | **26,094,944** |


![model arch](https://github.com/user-attachments/assets/7e17978a-13fb-4e9e-95fb-6f3ae0f96f02)


## 3. Training Details

The model was trained under the following settings:

*   **Device:** CPU (due to GPU unavailability)
*   **Loss Function:** CrossEntropyLoss
*   **Optimizer:** Adam (learning rate = 0.001)
*   **Learning Rate Scheduler:** ReduceLROnPlateau (patience=3, factor=0.1)
*   **Batch Size:** 32
*   **Number of Epochs:** 10
*   **Input Image Size:** 224x224
*   **Normalization:**  Images were normalized using `transforms.Normalize(mean=[0.485, 0.456, 406], std=[0.229, 0.224, 0.225])`

## 4. Results and Evaluation

The model achieved a **Best Test Accuracy: 96.83%**. The following metrics were used for evaluation:

## Classification Report
Demonstrates precision, recall, and F1-score for each class.

**Table 2: Classification Report**

| Class     | Precision | Recall | F1-score | Support |
| :--------- | :-------: | :-----: | :-------: | :-----: |
| ain       | 0.97      | 0.99    | 0.98      | 635     |
| al        | 0.98      | 0.98    | 0.98      | 403     |
| aleff     | 0.98      | 0.96    | 0.97      | 502     |
| bb        | 0.97      | 0.97    | 0.97      | 538     |
| dal       | 0.96      | 0.96    | 0.96      | 491     |
| dha       | 0.98      | 0.94    | 0.96      | 517     |
| dhad      | 0.99      | 0.97    | 0.98      | 501     |
| fa        | 0.93      | 0.95    | 0.94      | 587     |
| gaaf      | 0.98      | 0.94    | 0.96      | 512     |
| ghain     | 0.99      | 0.96    | 0.98      | 594     |
| ha        | 0.94      | 0.96    | 0.95      | 478     |
| haa       | 0.93      | 0.96    | 0.94      | 458     |
| jeem      | 0.97      | 0.95    | 0.96      | 466     |
| kaaf      | 0.97      | 0.96    | 0.97      | 533     |
| khaa      | 0.96      | 0.96    | 0.96      | 483     |
| la        | 0.99      | 0.98    | 0.98      | 524     |
| laam      | 0.98      | 0.99    | 0.98      | 550     |
| meem      | 0.98      | 0.98    | 0.98      | 530     |
| nun       | 0.98      | 1.00    | 0.99      | 842     |
| ra        | 0.96      | 0.96    | 0.96      | 498     |
| saad      | 0.97      | 0.97    | 0.97      | 569     |
| seen      | 0.98      | 0.98    | 0.98      | 492     |
| sheen     | 0.98      | 0.98    | 0.98      | 453     |
| ta        | 0.96      | 0.96    | 0.96      | 545     |
| taa       | 0.96      | 0.97    | 0.97      | 552     |
| thaa      | 0.94      | 0.96    | 0.95      | 530     |
| thal      | 0.98      | 0.96    | 0.97      | 475     |
| toot      | 0.96      | 0.97    | 0.96      | 538     |
| waw       | 0.98      | 0.99    | 0.98      | 412     |
| ya        | 0.98      | 0.98    | 0.98      | 517     |
| yaa       | 0.98      | 0.99    | 0.99      | 388     |
| zay       | 0.95      | 0.94    | 0.94      | 413     |
| **accuracy** |  **0.97** | | | **16526** |
| **macro avg** | 0.97      | 0.97    | 0.97      | 16526  |
| **weighted avg** | 0.97      | 0.97    | 0.97      | 16526  |


## Confusion Matrix Analysis for ArSLNet Performance

The confusion matrix provides a detailed, class-by-class breakdown of the ArSLNet model's predictive performance on the test dataset. The matrix, visualized as a heatmap, depicts the counts of true positives, false positives, and false negatives for each of the 32 ArSL classes. Key observations and their interpretations are presented below:

Dominant Diagonal: The strong intensity along the diagonal, indicated by high cell values, confirms the high overall accuracy reported in the classification report. This suggests that the model accurately classifies the majority of instances for most classes. Specifically, several classes exceed 500 correct predictions, demonstrating their robust recognition.

Class-Specific Misclassifications: Analysis of off-diagonal elements reveals potential areas of confusion. These misclassifications, while relatively infrequent compared to correct predictions, are valuable in identifying specific weaknesses in the model's ability to differentiate between certain gestures:

Visual Similarity: Some confusions likely arise from visual similarities between gestures. For instance, there seems to be some confusion between 'fa' and 'ghain'

"yaa" and "zay": The model can have some difficulty with this classes , indicating a potential need for improved feature representation or more training data for these specific gestures.

Support Imbalance Considerations: It's important to note that some classes have significantly larger support (number of samples) in the test set than others. As such, the magnitude of misclassifications must be interpreted relative to the support for each class. A small number of misclassifications in a high-support class might have a lesser overall impact than the same number of misclassifications in a low-support class.

Potential Impact on Real-World Applications: While the model achieves high accuracy overall, the observed misclassifications are critical to consider in the context of real-world ArSL communication systems. Even relatively infrequent errors could introduce ambiguity and misunderstandings, emphasizing the need for further model refinement to minimize these errors. The confusion matrix can be shown here:
![تنزيل](https://github.com/user-attachments/assets/9545707d-4799-4687-b4e8-7fd2393077a7)


## Loss and Accuracy Curves:** (See included image) shows the loss and accruacy during model training.
The training dynamics of the ArSLNet model are visualized in Figure 2, which presents plots of both training and testing loss and accuracy over the 10 epochs of training. Analyzing these curves provides insights into the learning behavior of the model and potential areas for further optimization.

Loss Curves:

The training loss decreases rapidly during the initial epochs, indicating that the model is quickly learning to fit the training data.

The testing loss exhibits a similar decreasing trend but plateaus at a higher value than the training loss. This consistent trend suggests that while the model may have room to fit the training data more closely (i.e. with more training or smaller learning rate), the test set validation indicates the model parameters have converged effectively.

Accuracy Curves:

The training accuracy increases sharply in the first few epochs and gradually approaches a high level, indicating that the model is learning to classify the training data correctly.

The testing accuracy mirrors this trend, achieving a high accuracy with a smaller increase each epoch.

The closeness of the test and training accuracies to convergence indicates that the model is generalizing well to unseen data.

Overall, the loss and accuracy curves show a model that was converging well, with no clear indications of overfitting or underfitting. The close performance on both test and training sets show this, although the model is more adapted to the training set.

Figure 2: Training and Testing Loss and Accuracy Curves. The loss curves (left) depict the decreasing loss on the training and test sets, and the accuracy curves (right) illustrate the increase in accuracy over the training epochs.

![تنزيل (1)](https://github.com/user-attachments/assets/31456a40-5c0a-41dd-8019-1022c265206c)

**Figure 1: Confusion Matrix of ArSLNet on the test dataset. Axes represent true and predicted labels, and the heatmap illustrates the frequency of classifications.**



## Integeration with OpenCV

We have to load the pre-trained pytorch model; specifally the [best_arsl_model.pth](https://drive.google.com/file/d/1rhVDFp0SIULFr2nnognrWhd5oWY8O_RO/view?usp=sharing) and [class_mapping](https://drive.google.com/file/d/1DhiRZHU23ME4AK6-c2jAPl0Xck73gXb5/view?usp=sharing)
## Testing the model with an image

![Screenshot 2025-02-07 193800](https://github.com/user-attachments/assets/2e7296de-9481-49ad-8e7f-90fe371012f6)

## Testing the model in-real time
![Screenshot 2025-02-07 185400](https://github.com/user-attachments/assets/011479ef-0961-4254-b7a5-dbdb66861043)



## 6. Conclusion

The presented ArSLNet CNN demonstrates strong performance in Arabic sign language recognition. This readme is a starting point to guide further research in this direction.
