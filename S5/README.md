# About Assignment S5
Consider a CNN network that can have either Group Normalization, Layer Normalization, Batch Normalization.
Compare how each of such networks converge.

# How it is achieved
It starts with taking the previous class' assignment code. 
The Net class __init__() is given an additional parameter that indicates the Normalization type.
For inputting the type, a new class NormType is created which extends the Enum class.

We provide 3 types - Group Normalization, Layer Normalization, Batch Normalization.
The train() function is changed to additionally have the logic for have L1 regularization on demand.
This flag is is used when invoking the training for Batch Normalization only.
4 is considered as the number of groups for Group Norm.

# Findings
All 3 of them have similar training loss and accuracy (too close for comfort).
However, the testing accuracy puts Batch Normalization better than Layer Norm, followed by Group Norm.

# Graphs
![image](https://user-images.githubusercontent.com/498461/217011301-73e37e96-911d-4eea-8117-818f66613b92.png)

# Misclassified Images
## Group Norm 
![image](https://user-images.githubusercontent.com/498461/217011576-86e2f1c7-411b-444b-972b-9ce087d47bf0.png)

## Layer Norm
![image](https://user-images.githubusercontent.com/498461/217011529-57931e7a-504e-43a9-9de8-b3e52125c52a.png)

## Batch Norm + L1 Regularization
![image](https://user-images.githubusercontent.com/498461/217011485-1786495d-89c7-4d19-a179-893ae29f7222.png)
