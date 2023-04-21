default_config = {
    "raw_file_path": "../input/raw/flickr8k/captions.txt",
    "clean_file_path": "../input/clean/flickr8k/captions.csv",
    "image_path": "../input/raw/flickr8k/Images",
    # "raw_file_path": "../input/raw/flickr30k/results.csv",
    # "clean_file_path": "../input/clean/flickr30k/captions.csv",
    # "image_path": "../input/raw/flickr30k/Images",
    "train_size": 0.8,
    "batch_size": 32,
    "num_workers": 4,

    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,

    "image_size": 224,
}
