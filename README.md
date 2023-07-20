# Breast-Cancer-Classification

# Preprocessing: 
The loaded images were labeled into classes 0 and 1 based on the folder structure, with 0 being ‘No Cancer’ and 1 being ‘IDC(+)’. The images were resized to 50*50 using inter_linear interpolation. The images were then loaded into a numpy array and shuffled for random distribution. The np array was then split into X for predictors and Y for label/target variable. The data was split into train and test samples in 70:30 ratio.the training data was in turn split into train and validation samples in the sample ratio as before. Since the dataset contains more than 200k records and most neural network models are RAM heavy, we have sub-sampled records from all the 3 datasets as inputs to our models.
# Cross-validation:
A smaller sample size was used to get the optimal parameters using the validation data. The models were then retrained on 50K training samples for 5-10 iterations and evaluated on 30K test samples.

# Classification:
A typical CNN model consists of a convolution layer followed by a max pooling layer. The more complex the problem, more layers need to be added. We have used a convolutional neural network model with ReLU activation function as our baseline model and have tried to compare this result with other models. Binary cross entropy is used as a loss function across all the models. The output of the final layer is sent through a dense layer with softmax activation function to get the probability values of the records belonging to a class. The different models that were implemented were:

Model 1:Baseline CNN model: The 3 Conv_2D layers with 32, 32 and 128 nodes in each layer
with padding in all the layers + 5 dense layers with ReLU activation function in all the layers and
softmax activation function in the last layer. He_uniform was used as a kernel function to
initialize weights in all the layers + 2 dropout layers with rate 0.3

Model 2: Modified CNN: 7 conv_2D layers + 5 dense layers + 4 dropout layers with rate 0.3.
Sigmoid activation function was used across layers and variance scaling was used for kernel
initializer.

Model 3: VGGNet: VGG16 layer, weights taken from pre-trained imageNet + Flatten layer + 4
dense layers with ReLU activation function and sigmoid in the last layer.

Model 4: ResNet: ResNet50 layer with weights initialized from imageNet + Flattening layer + 3
dense layer with ReLU activation function and dropout layers with 0.3 rate. Final dense layer
with softmax activation function for output.

Model 5: Inception_V3: Inception_v3 layer + dropout layer with rate set to 0.3 + flattening layer +
dense layer with softmax activation function. The input was reshaped to 75*75 for this model
rather than 50*50 that was used as default for every other model.
