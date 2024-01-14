import pandas as pd
import os 
import cv2
import numpy as np

OUTPUT_PATH = r'F:\tech_projects\arias\algae_detection\under_water\Algae_Detection_Demo\Outputs'

# Define the base directory where you want to search for .png files
base_directory = r'F:\tech_projects\arias\algae_detection\under_water\Algae_Detection_Demo\Data\augmented'  # Replace with your directory path

# List to store the paths of .png files
png_files = []

# Walk through the base directory and its subdirectories
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.lower().endswith('.png'):
            png_files.append(os.path.join(root, file))


'''
  Create a dataframe with three cols/attributes: 
  png_name (called 'file_name' in the metadata), label, png_path

  Example: 
    file_name: D20160521T164054_IFCB107_05202.png
    label: oxy
    png_path: gdrive/MyDrive/UTOPIA/Hisham/Example_dataset/five_groups/D20160521T164054_IFCB107_05202.png
'''

# Each row of the .csv file corresponds to a "manually" labeled image in the 
# dataset
csv_file = r'F:\tech_projects\arias\algae_detection\under_water\Algae_Detection_Demo\Data\augmented\data.csv'

# Define dataframe as the entire .csv file (all the rows are images)
df = pd.read_csv(csv_file)

# Define a function to split the file_path column
def split_file_path(row):
    file_name = os.path.basename(row['file_path'])
    png_path = os.path.dirname(row['file_path'])
    return pd.Series({'file_name': file_name, 'png_path': png_path})

# Apply the function to each row using df.apply
df[['file_name', 'png_path']] = df.apply(split_file_path, axis=1)

# Drop the 'file_path' column
df = df.drop('file_path', axis=1)

# Save the DataFrame to a CSV file
df.to_csv('Data/processed/data.csv', index=False)

'''
  Squares and pads images to 128x128 pixels to prepare them for the neural network
'''

def preprocess_input(image):

    fixed_size = 128 # Final image should be 128 x 128
    image_size = image.shape[:2] # Gets the (y_dim, x_dim) for each image

    # The ratio needed to make the longest side of the image 128 pixels
    ratio = float(fixed_size)/max(image_size)

    # Calculates the new size by multiplying each dimension by the ratio
    new_size = tuple([int(x*ratio) for x in image_size])

    # Resizes the image to the new size
    img = cv2.resize(image, (new_size[1], new_size[0]))

    # Calculates the possible padding needed for the x and y dimensions
    delta_w = fixed_size - new_size[1]
    delta_h = fixed_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    # Makes a black border of 128x128 pixels around the image, so either
    # dimension less than 128 would be padded to 128
    color = [0, 0, 0] # RGB = 0,0,0 -> Black
    rescaled_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return(rescaled_image)

# Holds just the indexed labelling of images in the sample
image_labels = df.label
# Verifies the proper full path for the images in the sample
filename = df['png_path']

# Preprocesses ALL the images in the sample, and compiles all the images into a dataset
# NOTE: This function is the meat of the program, so it may take awhile (the 
# last run took 53 min).
dataset = []
for filename, label in zip(df['png_path'],image_labels):
  image = cv2.imread(filename)
  rescaled_image = preprocess_input(image)
  dataset.append(rescaled_image)

# Creates a stacked 4D array from the dataset
# dataset_4d = (# of Images, x dim = 128, y dim = 128, # of color inputs in RGB = 3)
dataset_4d = np.array(dataset)

# Saves the preprocessed images so we don't need to re-process every time
np.save(os.path.join(OUTPUT_PATH,"fivegroups_dataset_4d_20000.npy"),dataset_4d)

