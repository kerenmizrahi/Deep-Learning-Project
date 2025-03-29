r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

1.
    A. $X$ is a $64\times 1024$ tensor while $Y$ is a $64\times 512$ tensor, therefore the Jacobian tensor will be $\displaystyle 64\times 512\times 64\times 1024$ .
    
    B. Yes, it is sparse due to the block-diagonal structure. While the Jacobian for each sample $\pderiv{\mat{y}_i}{\mat{x}_i}$ is dense, the full Jacobian $\pderiv{\mat{Y}}{\mat{X}}$ for the batch includes contributions only from corresponding input-output pairs in the batch. This means that elements relating outputs of one sample to inputs of another sample are zero. Thus, the block-diagonal structure of the Jacobian makes it sparse across the batch dimension
    
    C. No, by using the chain rule we get that $\delta\mat{X} = \pderiv{L}{\mat{Y}}*\pderiv{\mat{Y}}{\mat{X}} = \pderiv{L}{\mat{Y}}*\pderiv{\mat{X}\mat{W}^{T}}{\mat{X}} = \pderiv{L}{\mat{Y}}*\mat{W}^{T}$.
    which only requires matrix multiplication meaning we don't need to calculate the Jacobian.
2.
    A. The shape of the Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$ is determined by the dimensions of the output $\mat{Y}$ and the weights $\mat{W}$. Since $\mat{Y}$ has a shape of $64\times 512$ and $\mat{W}$ has a shape of $512\times 1024$, the Jacobian tensor would have the shape $64\times 512 \times 512 \times 1024$
    
    B. Yes, it is sparse because each output unit is only connected to a subset of the input units through the weight matrix $\mat{W}$. The Jacobian tensor $\pderiv{\mat{Y}}{\mat{W}}$ contains the partial derivatives of each output element with respect to each weight element. many rows will nullified completely as a result.

    C. No, by using the chain rule we get that $\delta\mat{W} = \pderiv{L}{\mat{Y}}*\pderiv{\mat{Y}}{\mat{W}} = \pderiv{L}{\mat{Y}}*\pderiv{\mat{X}\mat{W}^{T}}{\mat{W}} = \pderiv{L}{\mat{Y}}*\mat{X}^{T}$.  which only requires matrix multiplication meaning we don't need to calculate the Jacobian.

"""

part1_q2 = r"""
**Your answer:**
    Backpropogation is a method used to effectively calculate the gradients of each layer in a NN by using the chain rule. However, it isn't required in order to train a decent-based NN. Alternatively we could manually calculate the gradients of the prediction w.r.t each parameter. However, this approach isn't practical for deep NN's with many layers and parameters as it'd be extremely time consuming and error-prone.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # ====== YOUR CODE: ======
    wtsd = 0.01
    lr = 0.022
    reg = 0.02
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )


    # ====== YOUR CODE: ======
    wstd = 0.1
    lr_vanilla = 0.022
    lr_momentum = 0.003
    lr_rmsprop = 0.00025
    reg = 0.006
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1.
    The graphs match our expectations: 
    The graph without dropout overfits the data - reaches high train accuracy(\~95%) while the test accuracy is extremely low(\~23%) which is expected after 30 epochs.
    The graph with 0.4 dropout-rate has a smaller gap between the train accuracy and the test accuracy as a result of the dropout preventing overfitting by zeroing out neurons during training, which prevents the model from relying too much on specific features of the data. As a result we get a model that has lower train accuracy and higher test accuracy - a model that generalizes better.

2.
    We can clearly see that the high drop-out setting is underfitting when compared to the low drop-out setting. On the high setting the rate of neurons zeroing out is too high for the model to learn anything valuable. This behaviour is evident in the graph with the high drop-out rate where both the train and test accuracies are low, compared to the graph with the low drop-out rate.
    
        

"""

part2_q2 = r"""
**Your answer:**

Yes, it is possible for the test loss to increase for a few epochs while the test accuracy also increases. This can happen because test loss measures both the correctness and confidence of the model's predictions, whereas test accuracy only measures correctness.
For example, during training, the model might predict the true class with lower confidence (assigning a smaller probability), which increases the loss. At the same time, the model might improve its predictions on previously misclassified examples, leading to higher test accuracy.
"""

part2_q3 = r"""
**Your answer:**

1.
    Gradient descent is an optimization algorithm used to minimize a loss function by iteratively updating model parameters in the direction of the negative gradient of the loss. 
    Backpropagation is a technique for efficiently computing gradients using the chain rule. It propagates the gradient of the loss backward through the network layers to calculate the gradient of each parameter. To summarize, Backpropagation computes the gradients, while gradient descent uses these gradients to update the parameters.

2.
    Gradient Descent computes the gradient of the loss function with respect to the weights using the entire training dataset. This approach ensures a more accurate calculation of the gradient; however, it is computationally expensive for large datasets.
On the other hand, Stochastic Gradient Descent (SGD) computes the gradient of the loss function using a small mini-batch at each update step. This makes SGD both computationally efficient and memory-efficient, as it only requires storing the gradient of a small subset of the data at any given time.  

3.
    Stochastic Gradient Descent is used more often in deep learning practice for several reasons:
Computational Efficiency: SGD is computationally more efficient than Gradient Descent as it calculates gradients for only a single sample or a smaller batch of samples, rather than the entire dataset, at every iteration.
Escaping Local Minima: The inherent noise introduced by SGD, due to the high variance of gradients between samples, helps the model escape local minima and find better solutions.
Memory Efficiency: Deep learning datasets are typically very large and cannot fit entirely in memory. SGD is better suited for such cases as it allows incremental updates to model parameters, enabling training on large datasets.
Frequent Updates: By iterating over smaller batches, SGD updates the model parameters more frequently. This is particularly beneficial during the initial phases of training, where the model often underfits. These frequent updates can lead to faster convergence and improved generalization.

4.
    A. The approach of splitting the data into disjoint batches, performing multiple forward passes until all data is exhausted, and then doing a single backward pass on the summed loss is equivalent to gradient descent.
    $L(X) = L(X_1) + L(X_2) + ... + L(X_n)$ where $X_1, X_2, ..., X_n$ are the disjoint batches of data. The gradient of the total loss with respect to the model parameters is the sum of the gradients of the loss for each batch: $\frac{\partial L}{\partial \theta} = \frac{\partial L(X_1)}{\partial \theta} + \frac{\partial L(X_2)}{\partial \theta} + ... + \frac{\partial L(X_n)}{\partial \theta}$ Performing forward passes on each batch and accumulating their losses, followed by a single backward pass, effectively computes the same total gradient as processing the entire dataset in one pass, as is done in gradient descent. This method achieves the same result as GD but requires additional memory to store intermediate gradients or losses from each batch.

    B. This approach results in an out-of-memory error because, while the batches are small enough to fit in memory individually, the strategy requires retaining all intermediate results from the forward passes of each batch in memory until the backward pass is executed. In backpropagation, the gradients are calculated using the intermediate activations and weights from the forward pass. If you defer the backward pass until all batches are processed, the model must store all the intermediate activations and states for every batch until the gradients are computed. This accumulates memory usage across batches, eventually exceeding the available memory, leading to an out-of-memory error.
"""

part2_q4 = r"""
**Your answer:**

1.  
    Let's assume that we have a computational graph of $f$ with $n$ nodes, where each node represents a differentiable function $f_i$.

A. 
    To reduce the memory complexity for computing $f$ in forward mode AD, we could calculate each layer $f_i$ and its derivative $f_i'$ at the point $x_0$, discarding the intermediate values once their contribution to subsequent layers is complete. By doing so: we only store the current layer value and its derivative, which reduces the memory complexity to $\mathcal{O}(1)$.
   Each query of $f_i$ and $f_i'$ costs $\mathcal{O}(1)$, and since we traverse all $n$ layers sequentially, the total computation cost remains $\mathcal{O}(n)$.

B. 
    For backward mode AD, the chain rule is applied as: $\nabla f(x_0) = \nabla f_n(f_{n-1}(...f_1(x_0))) = \nabla f_n \cdot \nabla f_{n-1} \cdot ... \cdot \nabla f_1$ Backward mode starts from the final layer $f_n$ and propagates gradients back to the input $x_0$. By discarding intermediate results after their contributions to the gradient computation are complete, the memory complexity can also be reduced to $\mathcal{O}(1)$. Each query costs $\mathcal{O}(1)$, and the overall computational cost remains $\mathcal{O}(n)$.

2.
    Yes, for general directed acyclic graphs (DAGs), memory complexity can be reduced by calculating gradients of each node with respect to its inputs and discarding values that are no longer needed for subsequent nodes. For cyclic graphs, this is not possible because earlier computations may depend on outputs from later layers, creating circular dependencies. These techniques only apply to acyclic graphs like those in deep learning models.

3. In deep architectures, such as VGGs and ResNets, reducing memory complexity is crucial due to the large number of layers. While these techniques do not reduce the $\mathcal{O}(n)$ computational cost of backpropagation, they significantly reduce memory requirements by discarding intermediate results as soon as they are no longer needed. This is especially useful when training large models on hardware with limited memory, like GPUs.

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. High Optimization Error - refers to the gap between the population loss and the actual loss obtained during training. occurs when the optimization algorithm fails to find a good set of parameters for minimizing the loss function during training.
To reduce the error we can consider the following techniques:
- Increase Receptive Field: Use dilated convolutions, pooling, or larger kernel sizes in convolutional networks to ensure that the receptive field captures enough context from the input.
- Learning Rate Schedules: Use adaptive learning rate schedules or optimizers like Adam that adjust the learning rate during training.
- Tuning Hyperparameters: Adjust hyperparameters such as Learning Rate, Momentum and Weight Decay to improve the optimization error.

2. High Generalization error - the difference between the training loss and the population loss on unseen data. It indicates that the model fits the training data well but fails to generalize to new, unseen examples. This usually occurs when the model overfits the training data or when the training data distribution does not match the test data distribution.
To reduce the error we can consider the following techniques:
- Dropout: Introduce dropout layers to randomly deactivate neurons during training, improving generalization.
- Cross-Validation: By splitting the dataset into multiple training and validation subsets, you can ensure that the model performs well across different data splits, reducing population loss.
- Early Stopping: Stop the training when the validation loss stops decreasing
- Regularization Techniques: Use L2 regularization to penalize large weights and encourage simpler models.

3. High Approximation error - Measures the difference between the optimal population loss achievable by any model and the loss achievable by the chosen model architecture. A high approximation error indicates that the model lacks sufficient complexity to capture the true patterns in the data.
To reduce the error we can consider the following techniques:
- Increase Model Capacity: Add more layers or neurons to ensure that the model's architecture is capable of capturing the complexity of the data.
- Expand Receptive Field: use larger kernel sizes, larger filter sizes or deeper networks to capture broader patterns

"""

part3_q2 = r"""
**Your answer:**

FPR indicates that the model is incorrectly predicting positive instances when the true class is negative. This could happen for the following reasons:
- Data Distribution Mismatch: The distribution of the training data might differ from the test data, causing the model to incorrectly predict positives in regions where the true class is negative.
- Overfitting: The model may overfit the training data, assigning positive predictions to areas where the true class is negative due to overconfidence in specific patterns seen during training.
- Class Imbalance: When there are significantly more positive instances in the training data, the model might predict the majority class more often, leading to a higher FPR.

the exact same reasons stand for FNR, with the right adjustments.
"""

part3_q3 = r"""
**Your answer:**

1.
    In this case, false positives are costly because they result in many unnecessary expensive and high-risk further tests. Thus, the goal is to minimize the false positive rate (FPR) to reduce the number of patients incorrectly classified as sick.
However, false negatives are less critical here because a patient who is initially misclassified as healthy will develop non-lethal symptoms that can later confirm the diagnosis and allow for treatment.
To achieve this, we would select a point on the ROC curve that maximizes specificity (low FPR) and ensures a reasonably high true negative rate (TNR). This strategy minimizes unnecessary testing while tolerating some false negatives, as they do not have severe consequences in this scenario.

2.
    In this case, false negatives are extremely costly, as they result in missed diagnoses, potentially leading to high mortality rates. Therefore, the goal is to minimize the false negative rate (FNR) to ensure that as many cases as possible are caught early.
In this situation, false positives are less problematic because the additional testing is justified to save lives. As such, the selected point on the ROC curve would maximize sensitivity (high true positive rate, TPR) while tolerating a higher FPR.
The emphasis is on early detection and avoiding missed diagnoses at all costs, even if it means accepting a higher number of false positives.
"""


part3_q4 = r"""
**Your answer:**
MLPs are not well-suited for sequential data like text for the following reasons:
- No Temporal Awareness: MLPs treat inputs independently, ignoring the order of words, which is crucial for understanding sentence meaning (e.g., "not happy" vs. "happy not").
- Inability to Capture Long-Term Dependencies: Sentiment often depends on distant words in a sentence. MLPs lack the mechanisms to retain or utilize such dependencies.
- Loss of Structure: MLPs require flattening sequences, losing the inherent order and structure of the data

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer

    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.04
    weight_decay = 0.0001
    momentum = 0.01
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


1.  Each layer has $C_{out}\times(C_{in}\times F^2 + 1$ parameters, where $C_{out}$ is the number of output channels, $C_{in}$ is the number of input channels, and $F$ is the filter size.

Regular block:
- 1st $3\times3$ conv: $256 \times (256 \times 3 \times 3 + 1) = 590,080$
- 2nd $3\times3$ conv: $256 \times (256 \times 3 \times 3 + 1) = 590,080$

for a total of: $1,180,160$

Bottleneck block:
- 1st $1\times1$ conv: $64 \times (256 \times 1 \times 1 + 1) = 16,448$
- 2nd $3\times3$ conv: $64 \times (64 \times 3 \times 3 + 1) = 36,928$
- 3rd $1\times1$ conv: $256 \times (64 \times 1 \times 1 + 1) = 16,640$

for a total of: $70,016$

as seen above, the BottleNeck block has significantly less parameters in comparison with the regular block.

2.  The number of operations required for a convolutional layer is dependent on both the number of parameters (weights) and the size of the input tensor. The bottleneck block, requires significantly fewer operations compared to the regular block. This reduction in operational complexity arises from the lower number of parameters in the bottleneck architecture. Consequently, since the bottleneck block has fewer parameters, it also requires fewer operations to compute the output.

3.  

*Spatial:*
- Regular block: two convolution layers of 3x3 are used for a respective field of 5x5.
- BottleNeck block: a single 3x3 and two 1x1 convolotional layers are used for a respective field of 3x3.
In conclusion, the regular block combines the input better spatially.
   
*Across feature map:*
- Regular block: has more power in combining the input across feature maps, since it doesn't reduce the number of channels. It has more filters and therefore a wider variety of ways combine the channels.
- BottleNeck block: reduces the number of channels the input has, thus loosing some of the cross-channel information which was possible to capture beforehand.


"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

**1.** We can see that for both training and test set, all models with the same fixed value of 𝑘 eventually achieved similar accuracy.
While deeper networks initially had lower accuracy, they converged faster and caught up to the shallower ones.
The best training accuracy and the best test accuracy were produced by the 𝐿=4 model while 𝐿=2 was not much different.
A possible explanation for this behaviour can be that deeper NN have more parameters to learn, which makes them harder to train.
Another thing we can notice is that both 𝐿=4, 𝐿=2 models have overfitted to the training data as the difference between the train accuracy and test accuracy is pretty significant.
The problem could have been solved with regularization techniques such as dropout.

**2.** Yes, we can see in that for both K=32 and K=64 the network was not trainable for L=16. 
L=16 achieved significantly worse results - both train and test accuracies were around 10%, which suggests the network failed to learn anything useful.
There can be few reasons for this problem, one of them is:
Vanishing or Exploding gradients - as gradients propagate through more layers of the network, they may either diminish to near zero or grow exponentially large for the model to learn anything.

Two possible ways to resolve this problem are:
- Using activation functions like ReLU or LeakyReLU - helps prevent the gradients from vanishing as a result of the property of the functions.
- Using skip connections - a technique that allows gradients to flow more easily through the network.


"""

part5_q2 = r"""
**Your answer:**

In this experiment, we analyzed the impact of the number of convolutional filters in each layer. From the plots, we observed that for a fixed number of L layers, K=128 and K=64 achieved the highest test accuracies (approximately 65%). However, K=32 consistently resulted in the lowest test accuracy across all L values and required more epochs to converge. Furthermore, for K=32, as the network depth increased (higher L), there was a slight decline in test accuracy.
We also noted that higher L values led to fewer training epochs.

Compared to the first experiment, we observed that for both 
K=32 and K=64, L=4 achieved better test accuracy in Experiment 1.2 than in 1.1, and this was accomplished in fewer epochs. However, L=8 resulted in lower test accuracy in Experiment 1.2, while L=2 produced similar results across both experiments.

"""

part5_q3 = r"""
**Your answer:**

In this experiment, we analyzed the effect of the number of convolutional filters in each layer for `K=[64, 128]`.
From the graph we can see that `L=2` achieved the best results (around 65% test accuracy) with `L=3` coming as a close second, while 
`L=4` failed to learn. These results are consistant with previous experiments where we've seen that deeper
NN performed worse than the shallower ones.


"""

part5_q4 = r"""
**Your answer:**

In this experiment, we analized the effect of skip connections on the training and performance.
Starting with `K=32`, all three models were able to learn the data, `L16` was best with 70% test accuracy,
`L8` was close behind with around 65% test accuracy, while `L32` was the worst out of the three with around 60% test accuracy.
In comparison, in the first experiment using the CNN model with `L=16` failed to learn anything(and so would `L=32` if we ran it); 
`L=8` reached a similar test accuracy.

with `K=[64, 128, 256]`, all three models were able to learn the data, `L4` was best with around 70% test accuracy,
`L2` was close behind with similar test accuracy, while `L8` was the worst out of the three with around 60% test accuracy.
In comparison, in the third experiment `L=4` failed to learn anything(and so would `L=8` if we ran it); 
`L=2` reached a lower test accuracy of around 65% using the CNN model
"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
1. In img1, the model detected three of the dolphins but failed to classify them currectly. It classified two of the dolphins as "persons" with confidence scores of 0.47 and 0.90, while the third dolphin was classified as a "surfboard" with a confidence score of 0.37.
In img2, the model detected three dogs but misclassified them as other objects and failed to detect the cat entirely. Additionally, two of the dogs were misclassified as "cats" with confidence scores of 0.65 and 0.39, while only one dog was correctly classified as a "dog" with a confidence score of 0.50.

2. The model's failures can be attributed to the complexity of the images and/or overlapping objects.
In img1, two dolphins are overlapping, and there are water drops near the dolphin figures, which may add complexity to the image. Additionally, the dolphins are black against a lighter background, which might make classification more challenging.
In img2, the dogs and the cat are overlapping and close to each other. Also, the dogs’ ears may resemble cat ears, which might affect the model’s classification.
A suggestion to resolve these issues could be using advanced tools like the R-CNN family models as they are better at handling such scenarios (even though YOLO models are faster than R-CNN). Another suggestion could be manipulating the images, for example, adjusting the colors in the dolphins' image so they aren’t as dark.

3. To attack an object detection model like YOLO, we can use the Projected Gradient Descent (PGD) method. The process starts with a target image we want to fool. We add a small amount of noise to the image and adjust this noise step by step.
In each step, we check how the model’s predictions—like detected objects, bounding boxes, and classifications—change due to the noise. We use this feedback to modify the noise in a way that disrupts the model's accuracy while keeping the changes small enough that the image still looks the same to the human eye.
After repeating this process many times, we create a slightly altered image that appears unchanged to humans but causes the YOLO model to make mistakes, such as missing objects, misclassifying them, or drawing incorrect bounding boxes.

"""

part6_q2 = r"""
**Your answer:**

"""


part6_q3 = r"""
**Your answer:**

1.
- <u>Picture One</u> - Dog Behind Tree: This picture is an example of **occlusion**, as the tree hides parts of the dog's features (half of its face and body). Additionally, the colors behind the dog may make it harder to detect the dog's ear. Since this presentation lacks important features that a dog typically has, the model struggled to detect the dog as an object (or even objects overall).

- <u>Picture Two</u> - Dog as a Pirate: This picture is an example of **model bias**. While the model did detect the dog as an object, it typically does not see dogs wearing costumes, especially not as human characters. As a result, the model classified the dog as a "person" with a confidence score of 0.74.

- <u>Picture Three</u> - Jumping Dog: This picture is an example of **deformation**, as the dog appears blurry in the photo. Due to the dog jumping, distinct features of the dog became blurry, and parts of the dog moved, making some features look slightly different. For example, the dog's ears appear "pointier." These factors may lead to confusion and make it harder for the model to classify the object correctly. As a result, the model succeeded in detecting the dog as an object but misclassified it as a cat with a confidence score of 0.25.


"""

part6_bonus = r"""
**Your answer:**

The original image we provided to the model was of a polar bear rotated 90 degrees to the right. As a result of this rotation, the model failed to detect or correctly classify the object. One possible reason for this is that models like YOLO are highly sensitive to the orientation of objects, as they are typically trained on images where objects are aligned in common orientations. The model likely has limited exposure to "non-standard" orientations, such as the rotated polar bear. To address this, we manipulated the image by rotating it 90 degrees to the left. This adjustment made the polar bear appear in its natural, more typical orientation, which led the model to successfully detect and classify the polar bear with a confidence score of 0.75.

"""