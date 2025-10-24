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


SEPTEMBER 30th : 

The network has been built and the training process has started using the Python code called CODE_3.py. Fist results show good prediction capabilities. Next steps will be the following:
- Train the network more to increase the accuracy, plotiting loss function epoch after epoch.
- Test the accuracy of the network again.
- MOVE ALL THE SOFTWARE STRUCTURE ON GOOGLE COLAB (In order to operate coherently with Nicolae)
- Try to increase the resolution of input image and train the model again (but slower)
- test it. 

OCTOBER 1st : 

After training the network for 10 epochs (around 3 hours training on local GPU) it shows good results in classification: 87% accuracy.
The model has been trained on 512x512 px images, a little upscale with respect to the precedent versions (224x224px).
One can find the first results in RESULTS/GRAPHS/1_10_25

OCTOBER 2nd : 

Reorganizing files in a class architecture to clean the code and make it easier to adjust, train and expand for training, test, and the application process. 
All the working files are inside the AUESOME_rob_test folder. 
A new training set of 4800 images of 1024x1024px has been created and will be sampled during the next training session.
After strengthening the robustness of the network with new training sessions on a larger and higher-resolution dataset, we'll develop an application tool in a Python environment to test the network on ordinary images from outside our training set. Evaluating its efficiency, we'll decide what to do next. 
Apart from data augmentation, we could consider extending the classifier to videos with frame sampling and analysis of some frames from video sequences. 

OCROBER 8th :

Unfortunately the model seems to be overfitting on the training set. It shows excellent performances on the training set, good performances on the validation, but very bad performances on test set and random images. 
Necessary steps: 
- re train the network with bigger images ex.1024x1024px
- work on data augmentation (ex. lossy compression)✅

OCTOBER 23RD:

To better model performance and reduce overfitting, we started 
- increasing semantic variability in the training set by adding personal digital photographs; ✅
- apply lossy compression randomly on the train set to better network recongition on images with compression artefacts; ✅
- reduce network complexity by deactivating some nodes. ✅

ABOUT FILES

AUSOME_rob_test (FOLDER) : All the working files to train and test the network. 

AUSOME_rob_test.py : Current version of trainable CNN (update of CODE_3.py)

auesome_singlepic_eval.py : select a picture from your local disk and use the trained network to verify if the image is real or fake

CODE_1.py : is the first attempt at building a CNN based on ResNet-18.


CODE_2.py: adding Validation set and fixing summary command


CODE_3.py : Adding the preprocessing of the DFT and the DCT t the network code in the trasformation phase. 
VERY IMPORTANT: Conv layers of ResNet have been updated to have 9 layers instead of 3: DFT MAGNITUDE (R+B+G) + DFT PHASE (R+B+G) + DCT (R+B+G) 
In this way we might find a stonger pattern recognition power in the network.


make_subset.py : is a Python code to create a randomized subset of 10% of elements out of the original dataset and resize them to have the long edge of 512px.
Now we are working with 2400 real images and 2400 fake images for the training set.


data_preproc.py : is a Python code to compute the 3 channels DFT magnitude and phase and the DCT of a given image. 

plot_utilis.py : needed for the visualization of Loss and accuracy after training epochs

model_evual.py : code for testing the results of the network (currently 74% of accuracy)

show_predictions.py : visualizing real and fake images as recognized by the model.
