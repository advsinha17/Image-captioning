import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import string
import numpy as np
import os

CWD = os.path.join(os.path.dirname(__file__))


# Load embeddings from a file
def load_embeddings(embedding_path):
    embeddings_index = {}
    with open(embedding_path, 'r', encoding = 'utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype = 'float32')
            embeddings_index[word] = embedding
    return embeddings_index

# Create an embedding matrix
def create_embedding_matrix(tokenizer, vocab_size, embeddings_index, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# Preprocess captions
def preprocess_captions(data_path):
    caption_dict = {}

    with open(data_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        tokens = line.strip().split(',')
        if len(tokens) < 2:
            continue
        image_name, caption = tokens[0], ' '.join(tokens[1:])
        
        # Remove punctuation and lowercase the words
        caption = ''.join([char for char in caption if char not in string.punctuation])
        caption = ' '.join([word.lower() for word in caption.split() if len(word) > 1])
        
        # Add start and end tokens
        caption = '<start> ' + caption + ' <end>'
        
        # Group captions by image
        if image_name in caption_dict.keys():
            caption_dict[image_name].append(caption)
        else:
            caption_dict[image_name] = [caption]
    
    return caption_dict

# Build a vocabulary with a threshold on word frequency
def get_vocab(caption_dict, vocab_threshold=3):
    
    all_captions = ' '.join([' '.join(captions) for captions in caption_dict.values()])
    words = all_captions.split()
    word_counts = Counter(words)
    
    vocab = {k for k, v in word_counts.items() if v >= vocab_threshold}
    
    # Add unknown token
    vocab.add('<unk>')
    
    # Replace words occuring less than 3 times with unknown token
    for _, captions in caption_dict.items():
        for i, caption in enumerate(captions):
            caption = ' '.join([word if word in vocab else '<unk>' for word in caption.split()])
            captions[i] = caption

    return list(vocab)

# Get the maximum length among all captions
def get_max_length(caption_dict):
    max_length = 0
    for captions in caption_dict.values():
        for caption in captions:
            caption_length = len(caption.split())
            if caption_length > max_length:
                max_length = caption_length
    return max_length



if __name__ == '__main__':
    embeddings_path = os.path.join(CWD, 'glove.6B/glove.6B.200d.txt')
    embeddings_index = load_embeddings(embeddings_path)
    captions_path = os.path.join(CWD, 'data/captions.txt')
    caption_dict = preprocess_captions(captions_path)
    max_seq_length = get_max_length(caption_dict)
