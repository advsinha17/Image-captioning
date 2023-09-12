import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.text import Tokenizer
from utils import load_embeddings, create_embedding_matrix, preprocess_captions, get_vocab
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CWD = os.path.dirname(__file__)

class ImageCaptioningModel(Model):
    def __init__(self, 
                vocab_size, 
                max_seq_length, 
                embedding_matrix, 
                embedding_dim=200, 
                lstm_units=256, 
                dropout_rate=0.5):
        super(ImageCaptioningModel, self).__init__()
        
        self.lstm_units = lstm_units
        self.image_dense1 = layers.Dense(256, activation='relu')
        self.image_dense2 = layers.Dense(lstm_units, activation='relu')
        self.caption_embedding = layers.Embedding(input_dim = vocab_size, 
                                                  output_dim = embedding_dim, 
                                                  input_length = max_seq_length, 
                                                  weights = [embedding_matrix],
                                                  trainable = False,
                                                  mask_zero = True)
        self.lstm1 = layers.LSTM(lstm_units, return_sequences = True)
        self.lstm2 = layers.LSTM(lstm_units, return_sequences = True)
        self.lstm3 = layers.LSTM(lstm_units)
        self.dropout = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.add = layers.Add()
        
        # self.fc = layers.TimeDistributed(layers.Dense(vocab_size, activation='softmax', name='output'))
        self.fc = layers.Dense(vocab_size, activation='softmax', name='output')
        
    def call(self, inputs):
        image_input, caption_input = inputs

        ####### merged model
        image_features = self.dropout(image_input)
        image_features = self.image_dense1(image_features)
        caption_embedding = self.caption_embedding(caption_input)
        caption_embedding = self.dropout2(caption_embedding)
        lstm_output = self.lstm3(caption_embedding)
        merged = self.add([image_features, lstm_output])
        merged = self.image_dense2(merged)
        output = self.fc(merged)
        ##################################


        ####### Inject model
        # image_input, caption_input = inputs
        
        # # Process image features
        # image_features = self.image_dense1(image_input)
        # image_features = self.dropout1(image_features)
        
        # # Embed captions
        # caption_embedding = self.caption_embedding(caption_input)
        # caption_embedding = self.dropout2(caption_embedding)
        
        # # LSTM layers
        # lstm_output = self.lstm1(caption_embedding, initial_state=[image_features, image_features])
        # lstm_output = self.lstm2(lstm_output)
        # lstm_output = self.lstm3(lstm_output)
        
        # # Fully connected layer to get output
        # output = self.fc(lstm_output)

        return output
    
    def get_config(self):
        config = {
            "vocab_size": self.caption_embedding.input_dim,
            "max_seq_length": self.caption_embedding.input_length,
            "embedding_matrix": self.caption_embedding.get_weights()[0],
            "embedding_dim": self.caption_embedding.output_dim,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout.rate,
        }
        base_config = super(ImageCaptioningModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def build_graph(self, input_shape):

        images, captions = input_shape
        image_input = tf.keras.layers.Input(shape = images)
        caption_input = tf.keras.layers.Input(shape = captions)
        return tf.keras.Model(inputs = [image_input, caption_input], outputs = self.call([image_input, caption_input]))



        

        
    
        
if __name__ == "__main__":

    embeddings_path = os.path.join(CWD, 'glove.6B/glove.6B.100d.txt')
    embeddings_index = load_embeddings(embeddings_path)
    embedding_dim = 100
    captions_path = os.path.join(CWD, 'data/captions.txt')
    caption_dict = preprocess_captions(captions_path)
    tokenizer = Tokenizer()
    all_captions = [' '.join(captions) for captions in caption_dict.values()]
    vocab = get_vocab(caption_dict)
    tokenizer.fit_on_texts(vocab)
    vocab_size = len(vocab) + 1
    embedding_matrix = create_embedding_matrix(tokenizer, vocab_size, embeddings_index, embedding_dim)
    max_seq_length = 34
    model = ImageCaptioningModel(vocab_size=vocab_size, max_seq_length=max_seq_length, embedding_matrix=embedding_matrix, embedding_dim=embedding_dim)
    input_size = [(2048,), (max_seq_length,)]
    model.build_graph(input_size).summary()
    tf.keras.utils.plot_model(model.build_graph(input_size), to_file=os.path.join(os.path.dirname(__file__), 'model.png'), show_shapes=True)
