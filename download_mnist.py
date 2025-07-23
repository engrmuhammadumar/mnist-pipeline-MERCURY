# Save this as download_mnist.py
from keras.datasets import mnist
import numpy as np
import os
import lmdb
import caffe
from caffe.proto import caffe_pb2

def make_datum(img, label):
    return caffe_pb2.Datum(
        channels=1,
        width=28,
        height=28,
        data=img.tobytes(),
        label=int(label)
    )

def save_to_lmdb(images, labels, lmdb_path):
    map_size = images.nbytes * 10
    env = lmdb.open(lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for i in range(len(images)):
            datum = make_datum(images[i], labels[i])
            txn.put(f"{i:08}".encode('ascii'), datum.SerializeToString())
    env.close()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

os.makedirs("mnist_train_lmdb", exist_ok=True)
os.makedirs("mnist_test_lmdb", exist_ok=True)

save_to_lmdb(x_train, y_train, "mnist_train_lmdb")
save_to_lmdb(x_test, y_test, "mnist_test_lmdb")
