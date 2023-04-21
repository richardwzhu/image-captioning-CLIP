import albumentations as A
import cv2
import torch


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer,
                 transforms, config):
        self.image_filenames = image_filenames
        self.captions = list(captions)
        # Tokenize captions and pad/truncate them to max_length
        self.encoded_captions = tokenizer(
            self.captions, padding=True, truncation=True,
            max_length=config["max_length"]
        )
        self.transforms = transforms
        self.config = config

    # Returns a dictionary with tokenized captions, images, and captions
    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image_path = f"{self.config['image_path']}/{self.image_filenames[idx]}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        return len(self.captions)


# Transforms and augments images
def get_transforms(image_size):
    return A.Compose(
        [
            A.Resize(image_size, image_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True)
        ]
    )
