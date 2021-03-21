# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[train_result]: output/2021_03_21_train.png
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **drive_div4_speed20.py** which divides the output of the network by 4 and speeds up the simulation from 9 to 20 (smoother and faster output, overall this looks better). On the second level the original values coming from the model are much better.
* **model.h5** containing a trained convolution neural network 
* **writeup_report.md** (this file)
* **output/run1.mp4** the output for the first circuit with default values
* **output/run2_div4_speed20.mp4 **output using drive_div4_speed20.py - smoother, faster, nicer video.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
# To smooth out the continuous left/right steering:
# python drive_div4_speed20.py
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is mostly based on the architecture described at https://developer.nvidia.com/blog/deep-learning-self-driving-cars/ .

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 64, and a few fully connected layers. 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. Input images were cropped using the Keras layer Cropping2D, and there are two Dropout layers to prevent overfitting.

#### 2. Attempts to reduce overfitting in the model

As described previously, there are two Dropout layers (with 0.5 dropout rate), one between the convolution layers, and one between the fully connected (Dense) layers.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I've tried various batch sizes - lower ones tended to be much faster but less usable, higher ones were much slower.

Interestingly when I tried to run the model on my desktop computer (with a 1080 Ti GPU) I saw that GPU was utilized only between 15-20% - I was suspecting image processing, but since autonomous driving was not working on my computer I did not try to tune anything which could be related to this.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.  

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have convolutional layers which could detect lanes and other features correctly, and to have subsequent layers which could process this data into a correct steering value.

My first step was to use a convolution neural network model similar to the NVIDIA network used for its self driving cars here:  https://developer.nvidia.com/blog/deep-learning-self-driving-cars  .

I was thinking on transfer learning, but considering I was not trying to identify a single object here I did not try it.  

I did modify the mentioned network as the following:

- Crop 50/27 pixels at the top/bottom to preserve useful data only

- Due to the cropping there wasn't enough pixels left at the convolutional layers, I modified the subsampling for the first convolutional layer from (2,2) to (1,2)
- I've added two Dropout layers. I did not have problems with overfitting but considering training was slow I've included it in the beginning.

This was good enough for the first track (and it seems almost enough for the second one), so I did not try to alter this anymore.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes (difference from NVIDIA network is marked with **bold**):

- Input image (RGB - initial BGR was converted to RGB), with a size of 160x320
- **Crop layer - top 50 and bottom 27 pixels**
- **Lambda layer which normalizes the data between -1 and +1**
- Convolutional layer, depth 24, filter size 5x5, **subsampling 1,2** with ReLU activation

- Convolutional layer, depth 36, filter size 5x5, subsampling 2,2 with ReLU activation

- Convolutional layer, depth 48, filter size 5x5, subsampling 2,2 with ReLU activation

- **Dropout layer with 0.5 rate**

- Convolutional layer, depth 36, filter size 5x5, subsampling 2,2 with ReLU activation

- Convolutional layer, depth 48, filter size 5x5, subsampling 2,2 with ReLU activation

- Flatten layer to have a single dimension for the fully connected layers
- Fully connected layer (size: 100)
- **Dropout with 0.5 rate**
- Fully connected layer (size: 50)
- Fully connected layer (size: 10)
- Fully connected layer (size: 1)

#### 3. Creation of the Training Set & Training Process

The downloading and setup of the training set can be done via the command `bash img.sh` which downloads the two .zip files from my google drive which were used to train the model.

Since it is likely that keeping the car in the center is easier than recovering from when the car is drifting away I've created mostly "recovery" data.

I've captured many recovery scenarios driving from the edges back to the center, and there was much less center driving in the training data.

During training every frame was flipped vertically (along with negating the steering value), and the pictures from left/right "cameras" were also used with a +-1.0 steering bias (`leftright_baias` in the code). The bias was verified with the minimal data in `minitrain` though I think it is not the optimal value based on the end result.

I was initially thinking on using left/right pictures as additional depths for the input, but unfortunately the simulator provided only the center camera image. 



There are **9217** number of lines in the resulting training.csv, each containing 3 pictures. The center picture is flipped horizontally, therefore the final number of pictures is **4*9217 = 36868**.

I finatlly randomly shuffled the data set and put 20% of the data into a validation set. 

I was training for 4 epochs, as it can be shown below. The validation and training set errors are relatively close, there may be some overfitting on the training set but I consider based on the validation that the model is good enough.

![Training result][train_result]

#### 4. **Conclusions**

Judging by the end result I've concentrated on recovering too much. The model does very well in turns but not so much when going forward, it wiggles constantly. The parts from the training when the car reaches the middle but the car is still steering to the other side should be filtered out.

There should be more "normal driving" data where the car is in the middle with steering 0, this might remove the need for the previous filtering.

I've tried averaging the last N (N={3, 5}) steering values with no success. Dividing the steering value by 4 helped greatly as I did not want to re-train the model - this can be seen in the second video (`run2_div4_speed20.mp4`).

The model does surprisingly good in the second track, I will try to continue training the model (after loading the saved model) with the last part.

