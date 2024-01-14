import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import EarlyStopping

OUTPUT_PATH = r'F:\tech_projects\arias\algae_detection\under_water\Algae_Detection_Demo\Outputs'

# Each row of the .csv file corresponds to a "manually" labeled image in the 
# dataset
csv_file = r'F:\tech_projects\arias\algae_detection\under_water\Algae_Detection_Demo\Data\processed\data.csv'

# Define dataframe as the entire .csv file (all the rows are images)
df = pd.read_csv(csv_file)

# Holds just the indexed labelling of images in the sample
image_labels = df.label
# Verifies the proper full path for the images in the sample
filename = df['png_path']

# Loads the data from the file
dataset_4d = np.load(fr'{OUTPUT_PATH}\fivegroups_dataset_4d_20000.npy')

# Model/Data Parameters
num_classes = 5 # 5 Types of Phytoplankton
input_shape = (128, 128, 1) # (128 x pixels, 128 y pixels, 1 grayscale num)

'''
  Converts class vectors to binary class matrices

  Example: If image_labels[34] = 'Diatom', then we would create an array
           [0, 0, 0, 1, 0], where the index 3 represents
           'Diatom' and a 1 classifies the image as such

'''
image_label_dummys = pd.get_dummies(image_labels)

# Converts image to grayscale and not RGB -> Loses the last parameter
dataset_3d = dataset_4d[:,:,:,0] # Zeroes out the last parameter

'''
  Splits the data into training and testing data
  test_size = ratio of data that we use to evaluate model (1 - test_size is the ratio to train)
  random_state = seed for the random number generator
                 (ensures that multiple runthroughs have the same data initialization)
'''
x_train, x_test, y_train, y_test = train_test_split(dataset_3d, image_label_dummys, test_size=0.33, random_state=42)

'''
  Here we scale and verify the image measurements

  We want to convert the measurements in the input image to float32. 255 is
  the maximum value of a byte (the input feature's type before conversion), so we
  divide by 255 to scale the image between [0 and 1]. A value of 0-1 works well with
  the default learning rate and other hyperparameters in the model

  We also need to tune the images to have shape (128, 128, 1)
'''

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (128, 128, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

'''
    Here we build the neural architecture of the model
'''
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(4, 4), activation="relu"),
        layers.BatchNormalization(),  # Add Batch Normalization layer
        layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Dropout(0.25),  # Add Dropout layer
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        # layers.BatchNormalization(),  # Add Batch Normalization layer
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.1),  # Add Dropout layer
        layers.Dense(num_classes, activation="softmax"),
    ]
)

'''
    MODEL TRAINING
    Here we set the hyperparameters for training the model
'''
initial_learning_rate = .0005
lr_schedule = ExponentialDecay(
    initial_learning_rate,
    decay_steps=2000,  # Adjust this as needed
    decay_rate=0.5,  # Adjust this as needed
    staircase=False  # Set to True if you want a staircase decay
)

# Define early stopping criteria
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (e.g., validation loss)
    patience=8,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the model's weights to the best achieved during training
)

'''
  This cell trains the neural network with TensorFlow libraries and adjustable hyperparameters

  batch_size: number of images passed through training at one time (128)
  epochs: number of times the ENTIRE dataset will be passed through for training (15)
  validation_split: ratio of training data that will be used as a validation set for each epoch (10%)

  Note: Categorical Cross-entropy is a loss function that works well for single-label,
        multi-class categorization problems (like this one!)
'''

batch_size = 20
epochs = 50
callbacks = [early_stopping]

# keras.optimizers.Adam(learning_rate=lr_schedule)
# Training the model and checking accuracy on each epoch
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks)