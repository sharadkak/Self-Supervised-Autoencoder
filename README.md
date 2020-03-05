# Self-Supervised Autoencoder
This repository contains implementation of Autoencoder variants trained on YFCC00M dataset. An autoencoder was trained in self-supervised fashion as pretext task and later compared with vanilla autoencoder trained on same data. 

Three Autoencoder variants are trained.
1. Vanilla autoencoder trained using 224x224 images
2. Autoencoder trained on 9 tiles of 96x96
3. Autoencoder trained with an auxiliary task to solve jigsaw puzzle (https://arxiv.org/abs/1603.09246) where a classifier is attached to bottleneck representation of the autoencoder to predict the permutation number which is used to shuffle tiles of an image. 


## Implementation details
Self-supervised task used is a jigsaw puzzle, it involves creating 9 tiles of each image and then shuffle them using some permutation. Now Autoencoder is provided a batch of tiles to recontruct them. But we also attach a small layer classifier at the bottleneck who has to use compressed representation to predict the permutation used for that particular set of tiles, permutation indexes are the labels for classifier. 

This image may help to visualize it. 
