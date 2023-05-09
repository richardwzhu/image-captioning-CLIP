default_config = {
    "raw_file_path": "../input/raw/flickr30k/results.csv",
    "clean_file_path": "../input/clean/flickr30k/captions.csv",
    "train_image_path": "../input/raw/flickr30k/Images",
    "test_image_path": "../input/raw/flickr30k/Images",
    "image_retrieval_path": "../input/clean/flickr30k/captions.csv",
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
    "device": "cuda:0",

    # Image normalization parameters
    "image_size": 224,

    # Tokenizer parameters
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,

    # Encoder parameters
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

config_1 = {
    "raw_file_path": "../input/raw/flickr30k/results.csv",
    "clean_file_path": "../input/clean/flickr30k/captions.csv",
    "train_image_path": "../input/raw/flickr30k/Images",
    "test_image_path": "../input/raw/flickr30k/Images",
    "image_retrieval_path": "../input/clean/flickr30k/captions.csv",
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
    "device": "cuda:0",

    # Image normalization parameters
    "image_size": 224,

    # Tokenizer parameters
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,

    # Encoder parameters
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

config_2 = {
    "raw_file_path": "../input/raw/flickr30k/results.csv",
    "clean_file_path": "../input/clean/flickr30k/captions.csv",
    "train_image_path": "../input/raw/flickr30k/Images",
    "test_image_path": "../input/raw/flickr30k/Images",
    "image_retrieval_path": "../input/clean/flickr30k/captions.csv",
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
    "device": "cuda:0",

    # Image normalization parameters
    "image_size": 224,

    # Tokenizer parameters
    "text_tokenizer": "bert-base-uncased",
    "max_length": 200,

    # Encoder parameters
    "image_encoder": "resnet50",
    "text_encoder": "bert-base-uncased",
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

config_3 = {
    "train_raw_file_path": "../input/raw/flickr30k/results.csv",
    "test_raw_file_path": "../input/raw/flickr8k/captions.txt",
    "train_clean_file_path": "../input/clean/flickr30k/captions.csv",
    "test_clean_file_path": "../input/clean/flickr8k/captions.csv",
    "train_image_path": "../input/raw/flickr30k/Images",
    "test_image_path": "../input/raw/flickr8k/Images",
    "image_retrieval_path": "../input/clean/flickr8k/captions.csv",
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
    "device": "cuda:0",

    # Image normalization parameters
    "image_size": 224,

    # Tokenizer parameters
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,

    # Encoder parameters
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

config_4 = {
    "train_raw_file_path": "../input/raw/flickr8k/captions.txt",
    "test_raw_file_path": "../input/raw/flickr30k/results.csv",
    "train_clean_file_path": "../input/clean/flickr8k/captions.csv",
    "test_clean_file_path": "../input/clean/flickr30k/captions.csv",
    "train_image_path": "../input/raw/flickr8k/Images",
    "test_image_path": "../input/raw/flickr30k/Images",
    "image_retrieval_path": "../input/clean/flickr30k/captions.csv",
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
    "device": "cuda:0",

    # Image normalization parameters
    "image_size": 224,

    # Tokenizer parameters
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,

    # Encoder parameters
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

config_5 = {
    "flickr8k_raw_file_path": "../input/raw/flickr8k/captions.txt",
    "flickr30k_raw_file_path": "../input/raw/flickr30k/results.csv",
    "flickr8k_clean_file_path": "../input/clean/flickr8k/captions.csv",
    "flickr30k_clean_file_path": "../input/clean/flickr30k/captions.csv",
    "train_image_path": "../input/raw/combined/Images",
    "test_image_path": "../input/raw/combined/Images",
    "image_retrieval_path": "../input/clean/flickr30k/captions.csv",
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
    "device": "cuda:0",

    # Image normalization parameters
    "image_size": 224,

    # Tokenizer parameters
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,

    # Encoder parameters
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

config_6 = {
    "raw_file_path": "../input/raw/flickr30k/results.csv",
    "clean_file_path": "../input/clean/flickr30k/captions.csv",
    "train_image_path": "../input/raw/flickr30k/Images",
    "test_image_path": "../input/raw/flickr30k/Images",
    "image_retrieval_path": "../input/clean/flickr30k/captions.csv",
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
    "device": "cuda:0",

    # Image normalization parameters
    "image_size": 224,

    # Tokenizer parameters
    "text_tokenizer": "distilbert-base-uncased",
    "max_length": 200,

    # Encoder parameters
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
    "temperature": 5
}