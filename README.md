# AUESOME---Robustness-Test
CNN for Syntethic Images Classifier Through Spectral Analysis


ORIGINAL DATASET @ https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images?resource=download-directory


TIMELINE

- Build the dataset
- Choose the best CNN model to be implemented
- Extract DFT and Cosine Transform from every image of the dataset
- Start training the NN on the augmented dataset
- Test the model once trained
- Stress the model with data augemntation

Shall we think about three dymensional DFT? And try to compute the three channels differently?


ABOUT FILES

make_subset.py
is a Python code to create a randomized subset of 10% of elements out of the original dataset and resize them to have the long edge of 512px.
Now we are working with 2400 real images and 2400 fake images for the training set.

CODE_1.py
is the first attempt at building a CNN based on ResNet-18.
