# Assessment notes for week 4.
============================================================================================================
## Step 1
============================================================================================================
Target:
    Setup a basic Training  & Test Loop
    Try to achieve 99.4% test accuracy without any optimizations.
    There is no regard to number of parameters.
Results:
    Parameters: 6.3M
    Best Training Accuracy: 99.93
    Best Test Accuracy: 99.25
Analysis:
    Extremely Heavy Model because of the parameters... trains very slowly.
    Model is seen to overfit after the 10th epoch. 

============================================================================================================
## Step 2
============================================================================================================
Target:
    Make the model lighter, still without advanced optimizations.
    Try to achieve max test accuracy.
Results:
    Parameters: 10.7K
    Best Training Accuracy: 98.63
    Best Test Accuracy: 98.51
Analysis:
    Lighter model, and converged almost after the 3rd epoch.
    There wasn't any significant learning happening after that.
============================================================================================================
## Step 3
============================================================================================================
Target:
    Achieve test accuracy of 99.4.
    Keep the parameters under 10k.
    Put in all the optimizations such as Batch Normalization, Dropout layers after every convulution.
    Vary the bias between True/False to see the effect.
Results:
    Parameters: 9.08K
    Best Training Accuracy: 98.83
    Best Test Accuracy: 99.24
Analysis:
    Lighter model and is seen to fluctuate in terms of the loss twice in a 15 epoch training.
    Didn't find much effect of the bias being True / False.