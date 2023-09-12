import tensorflow as tf
import os
import pickle
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

INPUT_SIZE = (299, 299)
CWD = os.path.dirname(__file__)

def get_encodings(images_dir, model):
    model.summary()
    imgs_list = os.listdir(images_dir)
    encodings = {}
    for image_path in imgs_list:
        img = tf.keras.preprocessing.image.load_img(os.path.join(images_dir, image_path), target_size=(INPUT_SIZE + (3,)))
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.expand_dims(img, axis = 0)
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        encoding = model.predict(img)
        encoding = tf.reshape(encoding, encoding.shape[1])
        encodings[image_path] = encoding
    return encodings

if __name__ == '__main__':
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', input_shape = (INPUT_SIZE + (3,)))
    model = tf.keras.Model(base_model.input, base_model.layers[-2].output)
    encodings = get_encodings(os.path.join(CWD, 'data/Images'), model)
    with open(os.path.join(CWD, "encodings.pkl"), "wb") as f:
        pickle.dump(encodings, f)




