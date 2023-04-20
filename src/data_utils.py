import numpy as np
import pandas as pd


# Takes a raw caption data file and creates a new clean CSV
def preprocess(file_path, mode):
    if mode == 0:   # flickr8k
        df = pd.read_csv(file_path)
    else:   # flickr30k
        df = pd.read_csv(file_path, delimiter="|")
        df.drop(columns=[' comment_number'], inplace=True)
        df.columns = ['image', 'caption']

    df['id'] = df['image'].astype('category').cat.codes
    df['caption'] = df['caption'].str.lstrip()
    clean_path = f"../input/clean/{file_path.split('/')[3]}/captions.csv"
    df.to_csv(clean_path, index=False)
    print(f"Created clean csv file {clean_path}")
    return df


# Reads from a clean CSV and returns a train and validation df
def make_train_valid_dfs(csv_path, train_size):
    df = pd.read_csv(csv_path)
    image_ids = df.id.unique()
    train_ids = np.random.choice(image_ids, 
                                 size=int(train_size * len(image_ids)), 
                                 replace=False)
    valid_ids = [id_ for id_ in image_ids if id_ not in train_ids]
    train_df = df[df["id"].isin(train_ids)].reset_index(drop=True)
    valid_df = df[df["id"].isin(valid_ids)].reset_index(drop=True)
    return train_df, valid_df
