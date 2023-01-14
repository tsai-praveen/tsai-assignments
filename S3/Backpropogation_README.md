# Backpropogation using Excel

## 1. Network Architecuture
The document captures the experience of creating a backpropogation using an excel sheet. The Neural network looks like the below.
![image](https://user-images.githubusercontent.com/498461/212384169-399615a9-39d2-4e9a-8c2b-65ebfc0c1cde.png)

The network is accepting 2 inputs, and has a hidden and an output layer. The sigmoid function has been used as an activation function.

## 2. Formulae for backprop
The following depicts the formulae that were used in order to calculate the total loss, the gradients, and the new weights.
![image](https://user-images.githubusercontent.com/498461/212385158-01ba45f3-38f8-4081-b841-6e56bcdcd9de.png)

## 3. Calculating Table
The below is a depiction of the table showing how the loss changed when learning rate went from 0.2 (table 1), and 1 (table 2).
Kindly note for brievity rows 6 to 95 have been hidden.
![image](https://user-images.githubusercontent.com/498461/212386181-d2936d00-8ff9-4037-bdb4-3d6e091495d5.png)

## 4. Charts
The following chart depicts the change in loss over 100 epochs.
![image](https://user-images.githubusercontent.com/498461/212425230-47699598-8269-4f97-94c7-36f9928f7ded.png)

======================================================================================
PART - 2 of assignment
======================================================================================
# Addendum to the Jupyter Notebook

### 1. Network
The neural network has 5 convultion layers. All of the convolution layers have a kernel of size 3x3 with padding = 1 (except last).
Each convolution layer is followed by batch normalization. After every 2 conv layers, maxpooling is applied, followed by a dropout.
Global average pooling is finally applied on the input. 
The log softmax function is finally applied to get the final classification (this is better than softmax and gives realisitic outputs)

![image](https://user-images.githubusercontent.com/498461/212427336-59b17464-70c2-40d9-a67c-484f9a4494db.png)
The # of kernels have been reduced to keep the number of trainable parameters under 10K.

### 2. Training Data
The MNIST dataset has been utilsed for training / validation. The negative log likelihood is the loss function used. 

### 3. Training and Validation
The training and validation exercise have been performed for 20 epochs.
The start validation accuracy was around 97% and culminated at 99.14% in the final epoch.

![image](https://user-images.githubusercontent.com/498461/212428366-03e79d6f-6312-4693-afe7-f1368cd276e9.png)
