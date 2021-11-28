# ViT_pytorch
Coding a vision transformer from scratch using python's pytorch framework.
The Vision Transformer Architecture:

![img](dependencies/vit_arch.jpg)

Let's break down the subparts of the transformer...

* Initial patch embeddings: The input image is broken down into distinguishable patches of size nxn (n=16) and these patches are embedded using convolutional layers.
 
* Conversion to projected vectors: These patches are then converted to key, query and value projection vectors using linear layers

* Attention blocks

* Multi layer perceptron

* Final classification layer

