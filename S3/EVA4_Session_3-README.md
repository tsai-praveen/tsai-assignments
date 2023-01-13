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
