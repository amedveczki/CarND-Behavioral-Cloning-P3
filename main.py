import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, Conv2D, Dense, Flatten
from random import shuffle
import sklearn

lines = []

ROOT = "/opt/training/"
CSV = ROOT + "driving_log.csv"

with open(CSV) as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)


from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)



model = Sequential()


# set up lambda layer


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size//2):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = ROOT + batch_sample[0].split('\\')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(np.fliplr(center_image))
                angles.append(-center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

#ch, row, col = 3, 80, 320  # Trimmed image format

print("Creating sequential model")
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
#model.add(Lambda(lambda x: x/127.5 - 1.,
#        input_shape=(ch, row, col),
#        output_shape=(ch, row, col)))

model.add(Cropping2D(cropping=((50,50), (0,0)), input_shape=(160,320,3))) # TODO - check crop

model.add(Lambda(lambda x: (x / 255.0) - 0.5))#, input_shape=(160,320,3)))
#model.add(Flatten(input_shape=160,320,3)))
model.add(Conv2D(24,(5,5),subsample=(1,2),activation="relu"))
model.add(Conv2D(36,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),subsample=(2,2),activation="relu"))
model.add(Conv2D(64,(5,5),subsample=(1,1),activation="relu"))
model.add(Conv2D(64,(5,5),subsample=(1,1),activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


print("Compiling model...")
model.compile(loss='mse', optimizer='adam')


print("Fitting model...")
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples) * 2, validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples) * 2, 
    nb_epoch=5, verbose=1)

print("Saving model")

model.save('model.h5')
print("All done")
### print the keys contained in the history object
print(history_object.history.keys())
print(history_object)
import pickle
f = file("history.pickle", 'wb')
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
