import tensorflow as tf
import os

test_url = "http://download.tensorflow.org/data/iris_test.csv"
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

def download_data(destination="."):
    train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                               origin=train_dataset_url,
                                               cache_dir=destination)
                                               
    test_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(test_url),
                                              origin=test_url,
                                              cache_dir=destination)

    return train_dataset_fp, test_dataset_fp

if __name__ == "__main__":
    download_data()