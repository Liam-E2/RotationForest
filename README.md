# RotationForest
A python implementation of the Rotation Forest algorithm per https://arxiv.org/abs/1809.06705. 

# What is Rotation Forest?
Rotation Forest is an ensemble classification method similar to Random Forest, which addresses one of Random Forest's bigger weaknesses—decision trees can only partition the feature space orthogonally (perpendicular to the feature axes). To represent non-orthogonal boundaries, decision trees oftern overfit, requiring more splits than needed. Random Forest, which predicts the most common output of n trees trained on bootstrap samples, still may run into issues because of the base trees' gridlike decision structure. 

Rotation Forest tries to mitigate this by randomly "rotating" each tree. In practice, this is done by randomly splitting up the features of the model into some number of partitions, sampling from these partitions, and running PCA on the samples to get a "rotation matrix" (the eigenvector matrix, or principal components, depending on your background). By multiplying the original feature partitions and the rotation matrix, we rotate the features of the model, with randomness provided by the samples and random partitions. This means that each tree in the forest is "pointing in a different direction," allowing the ensemble to approximate curves more effectively than random forest.

# Example: Quadratic Data

![Train Data visualized](/mean0std1/Figure_1.png)

Here's a simple toy dataset with two quadratic curves + noise. We'll try fitting a Random Forest with stock settings and 200 trees, and a Rotation Forest with stock settings and 200 trees.

![Random Forest Predictions](/mean0std1/Rand.png)

The Random Forest gets a bit confused in the middle due to the non-gridlike boundary between the classes.

![Rotation Forest Predictions](/mean0std1/Rot.png)

Rotation Forest, while not perfect, "learns" the decision boundary much more accurately.
