# image-captioning-CLIP

Image captioning is a machine learning problem where the input received is an image and the output generated is a descriptive caption that is relevant to the picture and syntactically correct.

[OpenAI's CLIP](https://openai.com/research/clip) learns the relationship between an image and a full sentence describing it. CLIP uses Contrastive Language to associate similar images/captions in a latent space and disassociate different ones. CLIP has Zero-shot Learning capabilities so it is able to generalize on unseen labels and images without having to be trained with them. 

We built six different models based on CLIP from scratch using PyTorch. The models have varying image and text encoders, train and validation datasets, and creativity levels.
#### Experiments
```
0. Pre-trained ResNet50, Hugging Face DistillBERT, Flickr30k
1. Self-trained ResNet50 on CIFAR-10, Hugging Face DistillBert, Flickr30k
2. Pre-trained ResNet50, Hugging Face BERT, Flickr30k
3. Pre-trained ResNet50, Hugging Face DistillBERT, Train on Flickr30k, Train on Flickr8k
4. Pre-trained ResNet50, Hugging Face DistillBERT, Train on Flickr8k, Train on Flickr30k
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
