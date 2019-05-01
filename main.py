
import numpy as np
import os
from PIL import Image
import tensorflow as tf
    
import utils

if __name__ == "__main__":
    
    path = os.path.join(os.getcwd(),"Example")
    
    files = os.listdir(path)
    file = files[0]
    
    try:
        image = Image.open(os.path.join(path,file)).convert(mode = 'L')
    except IOError:
        print("cannot open file")
        
    pix = np.round(np.invert(utils.resize(np.asarray(image))) / 255)
    pix = pix.astype(np.uint8)
    
    letters = utils.split_letters(pix, False)
    
    Z1 = tf.placeholder(tf.float32)
    Z2 = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    saver = tf.train.import_meta_graph('./Trained Models/model_NN.meta')
       
    sess = tf.InteractiveSession()
    init = tf.local_variables_initializer()
    sess.run(init)
    saver.restore(sess, tf.train.latest_checkpoint('./Trained Models'))
    graph = tf.get_default_graph()

    W1_new = graph.get_tensor_by_name("W1:0")
    b1_new = graph.get_tensor_by_name("b1:0")
    W2_new = graph.get_tensor_by_name("W2:0")
    b2_new = graph.get_tensor_by_name("b2:0")
    W3_new = graph.get_tensor_by_name("W3:0")
    b3_new = graph.get_tensor_by_name("b3:0")


    for im in letters:
        im2vec = np.reshape(im, im.size, 1)[np.newaxis]
        X = im2vec.astype(np.float32)
        Z1 = tf.nn.sigmoid(tf.matmul(X, W1_new) + b1_new)
        Z2 = tf.nn.sigmoid(tf.matmul(Z1, W2_new) + b2_new)
        Y = tf.argmax(tf.nn.softmax(tf.matmul(Z2, W3_new) + b3_new),1)
        print(utils.idx2letter(Y.eval()[0]))