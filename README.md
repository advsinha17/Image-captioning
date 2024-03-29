## Image Caption Generator

Image Caption Generator is a deep learning project aimed at developing a tool that can automatically generate captions for images. It leverages computer vision and natural language processing techniques to create meaningful captions. Dataset used is [flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k). 

## Status

**Incomplete** - The project is currently incomplete. I am experimenting with different model architectures (as described in [this](https://arxiv.org/pdf/1703.09137.pdf) paper). The model I have trained in the `main.ipynb` file currently seems to overfit the training data.

Pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings are used.

## Prerequisites

* Python: 3.10
* Libraries: TensorFlow 2.x, NumPy, scikit_learn

## Project Structure

* `dataset.py`: Implements DatasetGenerator class which generates batches of inputs of the form ([image encoding, tokenized sequence], one-hot encoding of next word). 
* `test.ipynb`: Tests if the data generated by the DatasetGenerator is in the correct form.
* `extract.py`: Extracts the encodings of the images using an InceptionV3 network and stores them in `encodings.pkl`. 
* `model.py`: Defines the model used for image captioning.
* `utils.py`: Defines functions used to extract pre-trained word embeddings, clean the captions (remove punctuation, convert to lower case), and create the vocabulary.
* `main.ipynb`: Trains the model.

## To Do:

* Find and train optimal model for this task.
* Evaluation on BLEU score.
* Possibly use beam search to generate captions.







