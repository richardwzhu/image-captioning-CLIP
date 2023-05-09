import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from transformers import DistilBertTokenizer, BertTokenizer

from clip_utils import get_transforms
from data_utils import build_loaders, make_train_valid_dfs


# Train a model for a single epoch
def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    # Iterate through batches of data
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        # Update running loss for the epoch
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg)
    return loss_meter


# Run a validation epoch for a model
def valid_epoch(model, valid_loader, device):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    # Iterate through batches of data
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        # Update running loss for the epoch
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


# Track the loss for an epoch
class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


# Retrieves n captions that most resemble the given image
def get_captions(model, config, clean_file_path, image_path, n):
    device = config['device']
    model.eval()

    # Normalize given image
    transforms = get_transforms(config['image_size'])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms(image=image)['image']
    image = torch.tensor(image).permute(2, 0, 1).float()

    # Calculate the embedding for the given image
    with torch.no_grad():
        image_embedding = model.image_projection(model.image_encoder(image.to(device).unsqueeze(0)))

    # Create dataframe with all values
    train_df, valid_df = make_train_valid_dfs(clean_file_path, 0.8)
    df = pd.concat([train_df, valid_df])

    # Create a list of all captions from the dataframe
    captions = list(df.caption.values)

    # Tokenize all captions from dataframe
    if config["text_tokenizer"] == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizer.from_pretrained(config['text_tokenizer'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_tokenizer'])
    encoded_captions = tokenizer(
        captions, padding=True, truncation=True,
        max_length=config["max_length"]
    )

    # Split the input_ids and attention_mask for captions
    captions_dataset = list(zip(encoded_captions.input_ids, encoded_captions.attention_mask))

    top_k = []
    idx = 0
    for input_id, attention_mask in tqdm(captions_dataset):
        idx += 1
        # Calculate the embedding for a single caption
        caption_embedding = model.text_projection(model.text_encoder(
            input_ids=torch.tensor(input_id).to(device).unsqueeze(0),
            attention_mask=torch.tensor(attention_mask).to(device).unsqueeze(0)
        ))
        # Calculate the cosine similarity score between the image and the current caption
        cosine_similarity = torch.nn.functional.cosine_similarity(image_embedding, caption_embedding)

        # Check if the current caption has one of the n-highest cosine similarity scores and store it
        if len(top_k) == 0:
            top_k.append((cosine_similarity, idx))
        elif len(top_k) < n:
            for j in range(min(n, len(top_k))):
                if cosine_similarity > top_k[j][0]:
                    top_k.insert(j, (cosine_similarity, idx))
                    break
        else:
            for j in range(n):
                if cosine_similarity > top_k[j][0]:
                    top_k.insert(j, (cosine_similarity, idx))
                    top_k.pop()
                    break

    # Return captions based highest similarity scores
    top_k_captions = [captions[idx[1]] for idx in top_k]
    return top_k_captions


# Calculates the image embeddings for all images
def get_image_embeddings(model, config):
    # Create dataframe with images stored at 'image_retrieval_path'
    df, _ = make_train_valid_dfs(config['image_retrieval_path'], 1)

    if config["text_tokenizer"] == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizer.from_pretrained(config['text_tokenizer'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_tokenizer'])
    data_loader = build_loaders(df, tokenizer, mode="valid", config=config)

    model.eval()

    image_embeddings = []
    with torch.no_grad():
        # Encode and project image to create embeddings
        for batch in tqdm(data_loader):
            image_features = model.image_encoder(batch["image"].to(config['device']))
            image_embedding = model.image_projection(image_features)
            image_embeddings.append(image_embedding)
    return df, torch.cat(image_embeddings)


# Finds n-images that are most like the text query
def find_matches(model, df, image_embeddings, config, query, n=9):
    # Tokenize the text query
    if config["text_tokenizer"] == "distilbert-base-uncased":
        tokenizer = DistilBertTokenizer.from_pretrained(config['text_tokenizer'])
    else:
        tokenizer = BertTokenizer.from_pretrained(config['text_tokenizer'])
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(config['device'])
        for key, values in encoded_query.items()
    }

    # Encode and project text to create embeddings
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)

    # Normalize embeddings and calculate similarity between text and all images
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T

    # Retrieve images that most match query
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    image_filenames = list(df.image.values)
    matches = [image_filenames[idx] for idx in indices[::5]]

    # Display images
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{config['test_image_path']}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")

    plt.show()
