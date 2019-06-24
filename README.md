# FB Private and Secure AI Challenge

This is my personal repo in my undertaking of the lessons in my Udacity scholarship.

# Part 1
### Exercise 1.py
Notes: Simple creation of neural networks using tensors. Only one output vector is expected. And it's  [~0.317]

Activation function forces your values to be between [0, 1] since we want to represent values as probabilities.

# Part 2
### Exercise 1.py
Notes: One image is consisted of (64, 1, 28, 28). We need to transform this to (64, 28*28) since we want our image to be a 1D vector. 64 is basically the defined number of images in one batch. 1 color channel and 28x28 images. 

SoftMax changes the values of your vector to be distributed according to their values and when all are added we will get the answer of 1. Useful for representating probabilities distributed on n outputs.

Probabilities per image should equal to 1.

### Exercise 2.py
Notes:  The only requirement is that for a network to approximate a non-linear function, the activation functions must be non-linear. Here are a few more examples of common activation functions: Tanh (hyperbolic tangent), and ReLU (rectified linear unit).

There's a built in module for neural networks in PyTorch.

# Part 3
### Exercise 1.py
Notes: By minimizing this loss with respect to the network parameters, we can find configurations where the loss is at a minimum and the network is able to predict the correct labels with high accuracy. We find this minimum using a process called gradient descent. The gradient is the slope of the loss function and points in the direction of fastest change. To get to the minimum in the least amount of time, we then want to follow the gradient (downwards). You can think of this like descending a mountain by following the steepest slope to the base.

We measure loss through a loss function to calculate our prediction error.

### Exercise 2.py
Notes: Use autograd to perform backpropagation. Backpropagation is the process of going back to model to update the weights base on your calculated loss.

You need to forwarding, loss computing and backpropagation and hopefully minimize the error you produce in prediction. Clear the gradient always before forwarding to make sure you train your model right. Else, you will have the previous gradients come in your computations.

We make use of an optimizer to update our weights, specifying a learning rate and by the use of a stochastic gradient descent.

Epoch - number of times you ran the algorithm.

# Part 4
## Exercise 1
Fashion Classification exercise

# Part 5
## Exercise 1
Neural networks have a tendency to perform too well on the training data and aren't able to generalize to data that hasn't been seen before. this is called overfitting and it impairs inference performance. To test for overfitting while training, we measure the performance on data not in the training set called the validation set. 

## Exercise 2
Overfitting visualization. The graph shows that loss steadily goes down when you train but validation on test or foreign data shows no sign of a downhill slope. A clear sign of overfitting.

## Exercise 3
One technique to improve our model that is overfitting is to use dropout. We randomly drop nodes with a defined probability. Making it more flexible towards foreign data and avoid over generalization.

# Part 6
## Exercise 1
Saving and Loading Models

# Part 7
## Exercise 1
A common strategy for training neural networks is to introduce randomness in the input data itself. For example, you can randomly rotate, mirror, scale, and/or crop your images during training. This will help your network generalize as it's seeing the same images but in different locations, with different sizes, in different orientations, etc.

You'll also typically want to normalize images with transforms.Normalize. You pass in a list of means and list of standard deviations, then the color channels are normalized like so

input[channel] = (input[channel] - mean[channel]) / std[channel]

Subtracting mean centers the data around zero and dividing by std squishes the values to be between -1 and 1. Normalizing helps keep the network work weights near zero which in turn makes backpropagation more stable. Without normalization, networks will tend to fail to learn

# Part 8
## Exercise 1
Transfer learning is making use of pretrained models to solve problems they weren't train on. A sample is ImageNet, a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers. 