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
    "image_encoder_lr": 1e-4,
    "text_encoder_lr": 1e-5,
    "projection_head_lr": 1e-3,
    "weight_decay": 1e-3,
    "patience": 1,
    "factor": 0.8,
    "epochs": 2,
    "device": "cpu",

    # Image normalization parameters
    "image_size": 224,

    # Tokenizer parameters
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,

    # Econder parameters
    "image_encoder": "resnet50",
    "text_encoder": "distilbert-base-uncased",
    "pretrained": True,
    "trainable": True,

    # Projection head parameters
    "image_embedding": 2048,
    "text_embedding": 768,
    "projection_dim": 256,
    "dropout": 0.1,

    # CLIP parameters
    "temperature": 1
}
