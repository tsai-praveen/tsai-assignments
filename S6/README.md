# Session 6 readme.md file

## Assignment 6

    Run this network 

Links to an external site..
Fix the network above:

    change the code such that it uses GPU and
    change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)
    total RF must be more than 44
    one of the layers must use Depthwise Separable Convolution
    one of the layers must use Dilated Convolution
    use GAP (compulsory):- add FC after GAP to target #of classes (optional)
    use albumentation library and apply:
        horizontal flip
        shiftScaleRotate
        coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
    achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
    upload to Github
    Attempt S6-Assignment Solution.
    Questions in the Assignment QnA are:
        copy paste your model code from your model.py file (full code) [125]
        copy paste output of torchsummary [125]
        copy-paste the code where you implemented albumentation transformation for all three transformations [125]
        copy paste your training log (you must be running validation/text after each Epoch [125]
        Share the link for your README.md file. [200]

## Solution
The model.py has the code for the CNN as well as a new class for Depthwise Separable Convultion.
The network is a 4 layer convolution followed by an output layer.
The Depthwise Separable convolution layer is a 3x3 with padding 1, followed by a 1x1.
It is used in both the transition block as well as the convolution block.
A GAP layer is added at the end to flatten out the image.

The S6_Assignment.ipynb imports the network in model.py and trains it on the CIFAR10 dataset.
The mean and standard deviation of the CIFAR10 dataset is calculated for later use.
The class for performing Albumentations and providing it as a dataset is created.
It defines image augmentations such as graying, normalizations (based on mean / std above), flipping, shift scale, cut-out of training images and only normalization of the test images.

The network is instantiated and run for 100 epochs. 
At around epoch 66 we still that a steady 85% mark is breach and crosses and stays at 87% by epoch 100.