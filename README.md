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
