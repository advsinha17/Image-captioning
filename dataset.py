import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import random

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, caption_dict, features, tokenizer, max_length, vocab_size, batch_size=32, shuffle=True):

        self.caption_dict = caption_dict
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(caption_dict))
        
    def __len__(self):
        return int(np.ceil(len(self.caption_dict) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        list_images = [list(self.caption_dict.keys())[k] for k in indexes]
        X, y = self.__data_generation(list_images)

        return X, y
    
    
    def __data_generation(self, list_images):
        image_features, input_sequences, output_sequences = [], [], []

        # num = 0
        for i, image in enumerate(list_images):
            # print(i, image)
            for caption in self.caption_dict[image]:
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    # print(image, in_seq, out_seq)
                    # print(in_seq, out_seq, image)
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    # out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                
                    # print(out_seq.shape)
                    # num += 1
                    # image_name.append(image)
                    image_features.append(self.features[image])
                    input_sequences.append(in_seq)
                    output_sequences.append(out_seq)
        
        # output_sequences = np.array([pad_sequences(seq, maxlen=self.max_length) for seq in output_sequences])
        # output_sequences = pad_sequences(output_sequences, maxlen=self.max_length)
        # print("y_true shape:", np.array(output_sequences).shape)
        # print(num)
        return [np.array(image_features), np.array(input_sequences)], np.array(output_sequences)


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)