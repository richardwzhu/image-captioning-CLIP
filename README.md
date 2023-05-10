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
Follow the inference Jupyter notebooks.
#### Image Captioning

Code snippet from inference_0.ipynb
```
image_path = '../input/raw/flickr30k/Images/6155176.jpg'

n = 5
captions = get_captions(model, config, config['clean_file_path'], image_path, n)

for caption in captions:
    print(caption)
```

#### Image Retrieval

Code snippet from inference_0.ipynb
```
df, image_embeddings = get_image_embeddings(model, config)

query = "one dog sitting in grass"

find_matches(model, df, image_embeddings, config, query)
```

## Evaluation Results
#### Image Captioning
Reference image

![6155176](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/46dc4973-aa2d-4e56-ab97-9c6b9fe1c3ad)

Experiment 0 predictions:
```
A child is looking through a fence or gate .
Person watching a child as they play .
Two individuals on a beach jumping up with their arms and legs spread wide open .
A boy has climbed to the top of a slide .
Two children are looking through a telescope on a city street , and the boy is using a step ladder to see through the eyeglass .
```
Experiment 1 predictions:
```
The lights at night in the city .
a white dog with brown and black markings frolics in snow .
A man and a woman playfully fight and hang from support railings in a subway car .
people stop and stare at a large statue .
An officer in a reflective vest stands at the front of his van with his dog .
```
Experiment 2 predictions:
```
A baby in blue pants is sitting on a red slide .
A person on a ladder with a baby underneath .
A small child wearing light blue overalls sits on a red slide .
Girl with green tank top standing in the middle of a train track with multicolor train cars to the right .
Boy in yellow shirt sitting in yellow playground slide
```
Experiment 3 predictions:
```
A woman and baby are sitting in the window seat of a bus .
A group of people in hazmat suits are standing with a dummy on a stretcher .
A small boy wearing glasses and a hat is looking at something and smiling .
A little boy puts his face up to the uniquely-shaped window .
A little boy stands up next to a window and cries .
```
Experiment 4 predictions:
```
A small child at the bottom of a slide has hair sticking up .
A boy in red slides down an inflatable ride .
A small white dog is jumping over a branch on the ground covered with leaves .
A girl going down a slide .
A child with wild hair and sunglasses getting off a blue slide .
```
Experiment 5 predictions:
```
A woman watches her child play with a pair of sunglasses against a picture window .
Person watching a child as they play .
A laughing young boy is near a swimming pool .
A child is looking through a pretend telescope on playground equipment in front of a blue sky .
A child with wild hair and sunglasses getting off a blue slide .
```
Experiment 6 predictions:
```
A young person with red-hair and a green shirt peers into his telescope .
A little toddler trying to look through a scope but ca n't reach it .
A child looking through a telescope on a playground .
A boy has climbed to the top of a slide .
A man in a blue shirt is holding a young boy and looking through a telescope .
```

Using these reference captions:
```
little boy looking through a telescope on playground equipment with the sun glaring off the pole his hand is on .
A child is looking through a pretend telescope on playground equipment in front of a blue sky .
A little blond boy is looking through a yellow telescope .
A child looking through a telescope on a playground .
A kid with blond-hair using a telescope .
```

The BLEU score obtained for each experiment was:

<img width="331" alt="Screenshot 2023-05-10 at 12 46 03 PM" src="https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/a1e4e938-bda4-4d49-bcd9-96c793bf90be">

#### Image Retrieval
Text query: "one dog sitting in grass"

Experiment 0 images:

![0](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/942696fc-b938-41cc-ac00-3fdb65a3c5e8)

Experiment 1 images:

![1](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/2e9538cb-d5ce-4189-be77-bdc654f36f1f)

Experiment 2 images:

![2](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/4beb71c6-fcb9-44e4-a25f-3ca5c9076264)

Experiment 3 images:

![3](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/5bf38015-5dcc-411f-a3fa-12fbe6123f2c)

Experiment 4 images:

![4](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/31b2c2a1-3c79-4b40-9263-ee7355189f0f)

Experiment 5 images:

![5](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/3138b981-32f6-48dc-bee3-1a3c27833a91)

Experiment 6 images:

![6](https://github.com/richardwzhu/image-captioning-CLIP/assets/30671520/2cef761a-e5c1-4488-aec4-c6f43c7050a7)

## Observations
Takeaways:
- Models trained with more data are more robust and generalize better.
    - Experiment 5 achieved the best performance using a combined flickr8k and flickr30k dataset.
    - Experiment 1 had poor performance since its image encoder was trained on a much smaller dataset.
- Given the poor performance of experiments 3 and 4, flickr8k and flickr30k may have different linguistic and visual patterns/distributions that offset caption and image embeddings in the projection space.
- Larger architectures, like in experiment 2, do not necessarily have good performance, take longer to train, and reach convergence later.
- Model performance is noticeably impacted by the level of randomness/creativity in the responses. Since image captioning involves creative writing, increased diversity and novelty may be rewarded.

Next Steps:
- Acquire more training data and increase the topical diversity in both images and captions
- Train all models, especially those with larger architectures, for longer to improve model convergence and performance
