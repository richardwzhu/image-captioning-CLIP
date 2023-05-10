# image-captioning-CLIP

Image captioning is a machine learning problem where the input received is an image and the output generated is a descriptive caption that is relevant to the picture and syntactically correct.

[OpenAI's CLIP](https://openai.com/research/clip) learns the relationship between an image and a full sentence describing it. CLIP uses Contrastive Language to associate similar images/captions in a latent space and disassociate different ones. CLIP has Zero-shot Learning capabilities so it is able to generalize on unseen labels and images without having to be trained with them. 

We built six different models based on CLIP from scratch using PyTorch. The models have varying image and text encoders, train and validation datasets, and creativity levels.
#### Experiments
```
0. Pre-trained ResNet50, Hugging Face DistillBERT, Flickr30k
1. Self-trained ResNet50 on CIFAR-10, Hugging Face DistillBert, Flickr30k
2. Pre-trained ResNet50, Hugging Face BERT, Flickr30k
3. Pre-trained ResNet50, Hugging Face DistillBERT, Train on Flickr30k, Test on Flickr8k
4. Pre-trained ResNet50, Hugging Face DistillBERT, Train on Flickr8k, Test on Flickr30k
5. Pre-trained ResNet50, Hugging Face DistillBERT, Flickr30k + Flickr8k
5. Pre-trained ResNet50, Hugging Face DistillBERT, Increased “temperature”
```

We evaluated the quality of the machine-generated captions using the [BLEU metric](https://aclanthology.org/P02-1040.pdf) and examined our models' ability to retrieve relevant images given a text query.

[Project Slides](https://docs.google.com/presentation/d/1aLRRgcvadYfl0LNHZWuO7poyO-56RIeh63R_D2vCdk4/edit#slide=id.g2416b87c743_1_0)


## Repository Structure
```
├── input/
|   ├── clean/
|       ├── flickr8k
|           ├── captions.csv # Dataset mapping flickr8k captions to image file paths
|       ├── flickr30k 
|           ├── captions.csv # Dataset mapping flickr30k captions to image file paths
|   ├── raw/
|       ├── flickr8k
|           ├── captions.txt # Raw flickr8k captions
|           ├── Images/ # Folder of flickr8k images
|       ├── flickr30k 
|           ├── results.csv # Raw flickr30k captions
|           ├── Images/ # Folder of flickr30k images
|       ├── combined/
|           ├── Images/ # Folder of combined flickr8k and flickr30k images
├── models/ # Where lcoal PyTorch model checkpoints are stored
├── notebooks/ # Jupyter noteboks
|   ├── bleu.ipynb # Evaluting quality of machine-generated captions using BLEU metric
|   ├── exploration.ipynb # Initial data exploration
|   ├── inference_0.ipynb # Experiment 0 inference
|   ├── inference_1.ipynb # Experiment 1 inference
|   ├── inference_2.ipynb # Experiment 2 inference
|   ├── inference_3.ipynb # Experiment 3 inference
|   ├── inference_4.ipynb # Experiment 4 inference
|   ├── inference_5.ipynb # Experiment 5 inference
|   ├── inference_6.ipynb # Experiment 6 inference
|   ├── resnet50.ipynb # Building and training ResNet50 on CIFAR-10
├── scripts/ # Python scripts
|   ├── exp0.py # Script to run experiment 0
|   ├── exp1.py # Script to run experiment 1
|   ├── exp2.py # Script to run experiment 2
|   ├── exp3.py # Script to run experiment 3
|   ├── exp4.py # Script to run experiment 4
|   ├── exp5.py # Script to run experiment 5
|   ├── exp6.py # Script to run experiment 6
├── src/
|   ├── clip_utils.py # Everything related to the CLIP model
|   ├── config.py # Configuration dictionaries for all experiments
|   ├── data_utils.py # Functions to preprocess and transform the data
|   ├── train_eval_utils.py # Training and inference functions
```

## Environment Setup
1. Make sure your python version is `3.7+`
2. Install dependencies with `pip`:
```
pip install -r requirements.txt
```
## Training
Train CLIP model
```
python3 exp_0.py
```

With storing model artifacts in WandB
```
python3 exp_0.py --entity image-captioning-clip --project_name image-captioning-CLIP --exp_name exp_0
```

## Inference
#### Image Captioning
Reference image

![6155176](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/46dc4973-aa2d-4e56-ab97-9c6b9fe1c3ad)

Code snippet from inference_0.ipynb
```
image_path = '../input/raw/flickr30k/Images/6155176.jpg'

n = 5
captions = get_captions(model, config, config['clean_file_path'], image_path, n)

for caption in captions:
    print(caption)
```

Output
```
A child is looking through a fence or gate .
Person watching a child as they play .
Two individuals on a beach jumping up with their arms and legs spread wide open .
A boy has climbed to the top of a slide .
Two children are looking through a telescope on a city street , and the boy is using a step ladder to see through the eyeglass .
```

#### Image Retrieval

Code snippet from inference_0.ipynb
```
df, image_embeddings = get_image_embeddings(model, config)

query = "one dog sitting in grass"

find_matches(model, df, image_embeddings, config, query)
```

Output

![output](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/321f48ba-7d40-45b6-aabb-47ded256f243)

