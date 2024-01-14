import os
import shutil
import cv2
import pandas as pd


def augment_imgs(loaded_df, destination_dir):
    
    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    list_of_imgs = []
    for _, row in loaded_df.iterrows():

        destination_dir = os.path.join(destination_dir, {row['label']})
        # Check if the directory already exists
        if not os.path.exists(destination_dir):
            # If it doesn't exist, create the directory
            os.makedirs(destination_dir)
            print(f"Directory '{destination_dir}' created.")

        source_path = os.path.join(row['file_path'], image)
        file_name, _ = os.path.splitext(os.path.basename(source_path))
        destination_path = os.path.join(destination_dir, f"{file_name}_{0}.png")
        shutil.copy2(source_path, destination_path)
        list_of_imgs.append(destination_path)

        image = cv2.imread(source_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.flip(image, 0)
        destination_path = os.path.join(destination_dir, f"{file_name}_{1}.png")
        cv2.imwrite(destination_path, image)
        cv2.destroyAllWindows()
        list_of_imgs.append(destination_path)

        image = cv2.imread(source_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.flip(image, 1)
        destination_path = os.path.join(destination_dir, f"{file_name}_{2}.png")
        cv2.imwrite(destination_path, image)
        cv2.destroyAllWindows()
        list_of_imgs.append(destination_path)

        image = cv2.imread(source_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.flip(image, 0)
        image = cv2.flip(image, 1)
        destination_path = os.path.join(destination_dir, f"{file_name}_{3}.png")
        cv2.imwrite(destination_path, image)
        cv2.destroyAllWindows()
        list_of_imgs.append(destination_path)

        df = pd.DataFrame({
            'file_path': list_of_imgs,
            'label': row['label']
        })
        # Save the DataFrame to a CSV file
        df.to_csv('Data/augmented/data.csv', index=False)
        try:
            old_df = pd.read_csv('Data/augmented/data.csv', index_col=0)
            df = pd.concat([old_df, df])
            df.to_csv('Data/augmented/data.csv')
        except:
            df.to_csv('Data/augmented/data.csv')


destination_dir = r"F:\tech_projects\arias\algae_detection\under_water\Algae_Detection_Demo\Data\augmented"