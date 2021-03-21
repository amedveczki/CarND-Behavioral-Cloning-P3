import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Conv2D, Dense, Flatten, Dropout
from random import shuffle
import sys
import sklearn

lines = []

if sys.platform == "win32":
    #ROOT = "training/"
    ROOT = "minitrain/"
else:
    ROOT = "/opt/training/"
    #ROOT = "minitrain/"


CSV = ROOT + "driving_log.csv"

with open(CSV) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def correct_fname(fname):
    return ROOT + fname.split('\\')[-1]


# set up lambda layer

flip = False
flip_and_leftright = True
leftright_bias = 1

def generator(samples, batch_size=32):
    num_samples = len(samples)

    # Decrease internal batch size as we are providing more images per single line of sample
    if flip:
        batch_size //= 2
    elif flip_and_leftright:
        batch_size //= 4
        
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, int(batch_size)):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = correct_fname(batch_sample[0])
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                # flip if flip or flip + take left and right images
                if flip or flip_and_leftright:
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)
                    
                if flip_and_leftright:
                    # Based on "leftright_bias" include left/right pictures
                    left_image = cv2.imread(correct_fname(batch_sample[1]))
                    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)

                    right_image = cv2.imread(correct_fname(batch_sample[2]))
                    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                    images.append(left_image)
                    angles.append(center_angle + leftright_bias)
                    images.append(right_image)
                    angles.append(center_angle - leftright_bias)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

print("Creating sequential model")
model = Sequential()

# Add some cropping which cuts upper and lower part of the image
model.add(Cropping2D(cropping=((50,27), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# Subsampling here is modified compared to NVIDIA network as there wouldn't be enough pixels
model.add(Conv2D(24,(5,5),subsample=(1,2),activation="relu"))
model.add(Conv2D(36,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),subsample=(2,2),activation="relu"))

# Dropout to help overfitting
model.add(Dropout(0.5))
model.add(Conv2D(64,(5,5),subsample=(1,1),activation="relu"))
model.add(Conv2D(64,(5,5),subsample=(1,1),activation="relu"))
model.add(Flatten())
model.add(Dense(100))

# Another dropout to help overfitting
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

print("Compiling model...")
model.compile(loss='mse', optimizer='adam')

num_samples = len(train_samples)
num_val_samples = len(validation_samples)
if flip:
    num_val_samples *= 2
    num_samples *= 2
elif flip_and_leftright:
    num_val_samples *= 4
    num_samples *= 4
    

print("Fitting model...")
history_object = model.fit_generator(train_generator, samples_per_epoch =
    num_samples, validation_data = 
    validation_generator,
    nb_val_samples = num_val_samples, 
    nb_epoch=4, verbose=1)

print("Saving model")

model.save('model.h5')
print("All done")

# Save history for further processing if needed
import pickle
f = open("history.pickle", 'wb')
pickle.dump(history_object, f)
f.close

import matplotlib.pyplot as plt
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
