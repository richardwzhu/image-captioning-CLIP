import albumentations as A
import cv2
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import DistilBertConfig, DistilBertModel


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer,
                 transforms, config, mode):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        # Tokenize captions and pad/truncate them to max_length
        self.encoded_captions = tokenizer(
            self.captions, padding=True, truncation=True,
            max_length=config["max_length"]
        )
        self.transforms = transforms
        self.config = config
        if mode == "train":
            self.image_path = self.config['train_image_path']
        else:
            self.image_path = self.config['test_image_path']

    # Returns a dictionary with tokenized captions, images, and captions
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image_path = f"{self.image_path}/{self.image_filenames[idx]}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


class CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = ImageEncoder(
            image_encoder=config['image_encoder'],
            pretrained=config['pretrained'],
            trainable=config['trainable']
        )
        self.text_encoder = TextEncoder(
            text_encoder=config['text_encoder'],
            pretrained=config['pretrained'],
            trainable=config['trainable']
        )
        self.image_projection = ProjectionHead(
            embedding_dim=config['image_embedding'],
            projection_dim=config['projection_dim'],
            dropout=config['dropout']
        )
        self.text_projection = ProjectionHead(
            embedding_dim=config['text_embedding'],
            projection_dim=config['projection_dim'],
            dropout=config['dropout']
        )
        self.temperature = config['temperature']

    def forward(self, batch):
        # Getting image and text features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        # Getting image and text embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature,
            dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()


# Encodes images to a fixed size vector
class ImageEncoder(nn.Module):
    def __init__(self, image_encoder, pretrained, trainable):
        super().__init__()
        self.model = timm.create_model(
            image_encoder, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


# Encodes text to a fixed size vector
class TextEncoder(nn.Module):
    def __init__(self, text_encoder, pretrained, trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(text_encoder)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())

        for p in self.model.parameters():
            p.requires_grad = trainable

        # CLS token as sentence-level representation for classifcation
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


# Projects fixed size vectors into a new dimension
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


# Transforms and augments images
def get_transforms(image_size):
    return A.Compose(
        [
            A.Resize(image_size, image_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True)
        ]
    )


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
