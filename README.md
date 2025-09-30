# AUESOME---Robustness-Test
CNN for Syntethic Images Classifier Through Spectral Analysis


ORIGINAL DATASET @ https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images?resource=download-directory


GENERAL TIMELINE

- Build the dataset
- Choose the best CNN model to be implemented
- Extract DFT and Cosine Transform from every image of the dataset
- Start training the NN on the augmented dataset
- Test the model once trained
- Stress the model with data augemntation

Shall we think about three dymensional DFT? And try to compute the three channels differently?


SEPTEMBER 30th : STATE OF THE ART UPDATE

The network has been built and the training process has started using the Python code called CODE_3.py. Fist results show good prediction capabilities. Next steps will be the following:
- Train the network more to increase the accuracy, plotiting loss function epoch after epoch.
- Test the accuracy of the network again.
- MOVE ALL THE SOFTWARE STRUCTURE ON GOOGLE COLAB (In order to operate coherently with Nicolae)
- Try to increase the resolution of input image and train the model again (but slower)
- test it. 



ABOUT FILES

make_subset.py : is a Python code to create a randomized subset of 10% of elements out of the original dataset and resize them to have the long edge of 512px.
Now we are working with 2400 real images and 2400 fake images for the training set.

CODE_1.py : is the first attempt at building a CNN based on ResNet-18.


CODE_2.py: adding Validation set and fixing summary command


CODE_3.py : Adding the preprocessing of the DFT and the DCT t the network code in the trasformation phase. 
VERY IMPORTANT: Conv layers of ResNet have been updated to have 9 layers instead of 3: DFT MAGNITUDE (R+B+G) + DFT PHASE (R+B+G) + DCT (R+B+G) 
In this way we might find a stonger pattern recognition power in the network.


data_preproc.py : is a Python code to compute the 3 channels DFT magnitude and phase and the DCT of a given image. 

plot_utilis.py : needed for the visualization of Loss and accuracy after training epochs

model_evual.py : code for testing the results of the network (currently 74% of accuracy)

show_predictions.py : visualizing real and fake images as recognized by the model.
