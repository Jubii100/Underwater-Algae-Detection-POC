import os
import pandas as pd

def collect_training_images(directory, species_list, img_num):
    for search_string in species_list:
        for root, dirs, _ in os.walk(directory):
            for dir_name in dirs:
                if search_string.lower() in dir_name.lower():
                    dir_path = os.path.join(root, dir_name)
                    # print(dir_path)
                    for _, _, files in os.walk(dir_path):
                        list_of_imgs = []
                        for file in files:
                            # print(os.path.join(dir_path, file))
                            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', 'svg')):
                                list_of_imgs.append(os.path.join(dir_path, file))
                            if len(list_of_imgs) == img_num:
                                # return list_of_imgs
                                df = pd.DataFrame({
                                    'file_path': list_of_imgs,
                                    'label': search_string
                                })

                                # Save the DataFrame to a CSV file
                                df.to_csv('Data/loaded/data.csv', index=False)
                                try:
                                    old_df = pd.read_csv('Data/loaded/data.csv', index_col=0)
                                    df = pd.concat([old_df, df])
                                    df.to_csv('Data/loaded/data.csv')
                                except:
                                    df.to_csv('Data/loaded/data.csv')

                                break

                        print(f"{search_string} data loaded successfully")
    print(f"All data loaded successfully")