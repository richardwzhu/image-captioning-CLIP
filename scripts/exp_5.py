import itertools
import pandas as pd
import torch
import wandb

from transformers import DistilBertTokenizer

import sys
sys.path.append('../src')
from data_utils import preprocess, make_train_valid_dfs, build_loaders
from config import config_5
from clip_utils import CLIPModel
from train_eval_utils import train_epoch, valid_epoch


project_name = 'image-captioning-CLIP'
exp_name = 'exp_5'
config = config_5
device = config['device']
print(f'{project_name=}\n{exp_name=}\n{device=}')
print(f'{config=}')

preprocess(config['flickr8k_raw_file_path'], 0)
preprocess(config['flickr30k_raw_file_path'], 1)
flickr8k_train_df, flickr8k_valid_df = make_train_valid_dfs(
    config["flickr8k_clean_file_path"], 0.8)
flickr30k_train_df, flickr30k_valid_df = make_train_valid_dfs(
    config["flickr30k_clean_file_path"], 0.8)
train_df = pd.concat([flickr8k_train_df, flickr30k_train_df])
valid_df = pd.concat([flickr8k_valid_df, flickr30k_valid_df])
tokenizer = DistilBertTokenizer.from_pretrained(config['text_tokenizer'])
train_loader = build_loaders(train_df, tokenizer, mode="train", config=config)
valid_loader = build_loaders(valid_df, tokenizer, mode="train", config=config)

# run = wandb.init()
# artifact = run.use_artifact(f'richzhu/{project_name}/{exp_name}:latest',
#                             type='model')
# artifact_dir = artifact.download()
# run.finish()

run = wandb.init(project='image-captioning-CLIP', config=config)

model = CLIPModel(config)
# model.load_state_dict(torch.load(f'{artifact_dir}/{exp_name}.pt'))
# model.to(device)

params = [
    {"params": model.image_encoder.parameters(),
     "lr": config['image_encoder_lr']},
    {"params": model.text_encoder.parameters(),
     "lr": config['text_encoder_lr']},
    {"params": itertools.chain(
        model.image_projection.parameters(), model.text_projection.parameters()
    ), "lr": config['projection_head_lr'],
        "weight_decay": config["weight_decay"]
    }
]
optimizer = torch.optim.AdamW(params, weight_decay=0.)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=config['patience'], factor=config['factor']
)
step = "epoch"

best_loss = float('inf')
for epoch in range(config['epochs']):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler,
                             step, device)
    model.eval()
    with torch.no_grad():
        valid_loss = valid_epoch(model, valid_loader, device)

    if valid_loss.avg < best_loss:
        best_loss = valid_loss.avg
        torch.save(model.state_dict(), f'../models/{exp_name}.pt')
        artifact = wandb.Artifact(exp_name, type='model')
        artifact.add_file(f'../models/{exp_name}.pt')
        run.log_artifact(artifact)
        print("Saved Best Model!")

    lr_scheduler.step(valid_loss.avg)

run.finish()
