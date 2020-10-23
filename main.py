import tensorflow as tf
import numpy as np
from RANCC import RANCC_Classifier
import keras
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer




NUM_WORDS = 10000
INDEX_FROM = 3
maxlen = 50
num_classes = 2
num_patterns = int(0.5*maxlen)

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
np.load = np_load_old
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test =  keras.utils.to_categorical(y_test, num_classes=num_classes)
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test =  sequence.pad_sequences(x_test, maxlen=maxlen)


BATCH_SIZE = 64
embedding_size = 128
with tf.variable_scope('c') as scope:
    rancc_cls = RANCC_Classifier(
        sequence_length=x_train.shape[1],
        num_classes=y_train.shape[1],
        vocab_size=NUM_WORDS,
        embedding_size=embedding_size,
        lstm_units=embedding_size,
        batch_size=BATCH_SIZE,
        num_patterns = num_patterns)

    

    
# Define Training procedure
#===============================classifier optimizer ==========================
global_step = tf.Variable(0, name="global_step", trainable=False)
with tf.device('/gpu:0'):
    global_step1 = tf.Variable(0, name="global_step1", trainable=False)
    optimizer= tf.train.AdamOptimizer(rancc_cls.learning_rate)
    gradients,varibales = zip(*optimizer.compute_gradients(rancc_cls.cost))
    gradients,_ = tf.clip_by_global_norm(gradients,1.0)
    train_op=optimizer.apply_gradients(zip(gradients,varibales))
    
    
    
init_op = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True))
sess.run(init_op)    

def epoch_update(x_batch, y_batch, sess):
    """
    A single training step
    """

    feed_dict = {
        rancc_cls.input_x: x_batch,
        rancc_cls.input_y: y_batch,
        rancc_cls.training:1,
        rancc_cls.learning_rate:0.0001
        }        
    _, _, loss, accuracy = sess.run(
        [train_op, global_step, rancc_cls.cost, rancc_cls.accuracy],
        feed_dict)            

    return loss, accuracy

    
tot_batch = int(x_train.shape[0]/BATCH_SIZE)
print("Started training ...")
for epoch in range(1170):

    loss_cls = 0
    acc_cls = 0
    indices = np.arange((x_train.shape[0]))
    np.random.shuffle(indices)    
    x_train = x_train[indices]
    y_train = y_train[indices]
    for index in range(int(x_train.shape[0]/BATCH_SIZE)):
        x_batch = x_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
        y_batch = y_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]                                   

        loss, accuracy = epoch_update(x_batch, y_batch,sess)
        print("Accuracy ", accuracy)
        loss_cls += loss
        acc_cls += accuracy

    print("Epoch ", epoch)
    print("Classisifer_loss {:g},Accuracy {:g}".format(loss_cls/tot_batch,acc_cls/tot_batch))
