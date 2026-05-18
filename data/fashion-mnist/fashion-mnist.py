import numpy as np
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from keras.datasets import fashion_mnist

def main():
    os.makedirs("./data/fashion-mnist/train", exist_ok=True)
    
    (train_images, train_labels), (val_images, val_labels) = fashion_mnist.load_data()
    
    train_images = train_images.astype(np.float32)
    
    train_labels = train_labels.flatten().astype(np.float32)
    train_labels = tf.one_hot(train_labels, depth=10).numpy()
    
    train_images.tofile("./data/fashion-mnist/train/train_images.mat")
    train_labels.tofile("./data/fashion-mnist/train/train_labels.mat")
    
    val_images = val_images.astype(np.float32)
    
    val_labels = val_labels.flatten().astype(np.float32)
    val_labels = tf.one_hot(val_labels, depth=10).numpy()
    
    val_images.tofile("./data/fashion-mnist/train/val_images.mat")
    val_labels.tofile("./data/fashion-mnist/train/val_labels.mat")
    
if __name__ == "__main__":
    main()