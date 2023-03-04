import tensorflow as tf

def gather_prob(tensor, indices):
    shape = (tensor.get_shape().as_list())
    flat_first = tf.reshape(tensor, [shape[0] * shape[1]] + shape[2:])
    indices = tf.convert_to_tensor(indices)
    offset_shape = [shape[0]] + [1] * (indices.shape.ndims - 1)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    output = tf.gather(flat_first, indices + offset)

    return output


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def weight_variable(shape, nam):
    initializer = tf.truncated_normal_initializer(
        dtype=tf.float32, stddev=0.001)

    return tf.get_variable(nam, shape, initializer=initializer, dtype=tf.float32)


def sample_without_replacement(logits, K):
    z = -tf.log(-tf.log(tf.random_uniform(tf.shape(logits), 0, 1)))
    va, indices = tf.nn.top_k(logits + z, K, sorted=False)

    return va, indices


def gather_emb(values, indices):
    row_indices = tf.range(0, tf.shape(values)[0])[:, tf.newaxis]
    row_indices1 = tf.tile(row_indices, [1, tf.shape(indices)[-1]])
    indices2 = tf.stack([row_indices1, indices], axis=-1)

    return tf.gather_nd(values, indices2)


def bias_variable(shape, nam):
    initializer = tf.constant_initializer(0.0)

    return tf.get_variable(nam, shape, initializer=initializer, dtype=tf.float32)


def return_topn(h_fc1, n):
    values, indices = tf.nn.top_k(h_fc1, k=n, sorted=False)

    return values, indices


class RANCC_Classifier(object):
    """
    RANCC: Rationalizing Neural Networks via Concept Clustering
    Network structure: embedding layer > Rationale extraction > LSTM classifier > Concept clustering and softamx
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, lstm_units, batch_size, num_patterns, lambda_val=0.08):
        with tf.device('/gpu:0'):
            #            tf.set_random_seed(2)
            self.batch_size = batch_size
            self.input_x = tf.placeholder(
                tf.int32, [batch_size, sequence_length], name="input_x")
            self.input_y = tf.placeholder(
                tf.float32, [batch_size, num_classes], name='input_y')
            self.learning_rate = tf.placeholder(tf.float32)
            self.lstmUnits = lstm_units
            self.maxSeqLength = num_patterns
            self.lambda_val = lambda_val
            self.training = tf.placeholder(tf.int32)
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            self.W = tf.Variable(tf.random_uniform(
                [vocab_size, embedding_size], -1.0, 1.0), name="W_embedding")
            self.embdded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embdded_chars_expanded = tf.expand_dims(self.embdded_chars, 3)
            self.W_concept = tf.Variable(tf.random_uniform(
                [num_classes, embedding_size], -1.0, 1.0), name="W_concept")
            W_conv1 = weight_variable([1, 25, 1, 32], "wc1")
            b_conv1 = bias_variable([32], "bc1")
            self.h_conv1 = tf.nn.relu(
                conv2d((self.embdded_chars_expanded), W_conv1) + b_conv1)
            self.aggregation = tf.reduce_sum(self.h_conv1, 3)
            self.sampling_prob = tf.nn.softmax(tf.reduce_sum(
                self.aggregation, 2)/tf.sqrt(tf.cast(embedding_size, tf.float32)), name="sampling_prob")
            self.sampling_prob = tf.layers.dropout(self.sampling_prob, self.dropout_keep_prob)

            _, self.ind = tf.cond(self.training > 0, lambda: sample_without_replacement(
                self.sampling_prob, self.maxSeqLength), lambda: return_topn(self.sampling_prob, self.maxSeqLength))

            self.sampled_prob = gather_prob(self.sampling_prob, self.ind)
            self.rationale_emb = gather_emb(self.embdded_chars, self.ind)
            self.em = tf.reduce_mean(self.rationale_emb * tf.reshape(
                (self.sampled_prob), [-1, self.maxSeqLength, 1]), 1, name="lassfsfsft")
            self.mean_emb = tf.reduce_mean(
                self.rationale_emb, 1, name="rationale_mean")
            self.rationale = self.rationale_emb * \
                tf.reshape((self.sampled_prob), [-1, self.maxSeqLength, 1])

            self.lstmCell = tf.contrib.rnn.LSTMCell(
                self.lstmUnits, name="LSTMCELL")
            self.value, self.lstm_out = tf.nn.dynamic_rnn(
                self.lstmCell, self.rationale, dtype=tf.float32)

            self.last = tf.reduce_sum(
                self.value * tf.reshape((self.sampled_prob), [-1, self.maxSeqLength, 1]), 1, name="last")

            self.weight = tf.Variable(tf.random_normal(
                [self.lstmUnits, num_classes], stddev=0.01), name="w_att1")
            self.bias = tf.Variable(tf.random_normal(
                [num_classes], stddev=0.01), name="b_att1")
            self.scores1 = (tf.matmul(self.last, self.weight) + self. bias)
            self.soft1 = tf.nn.softmax(
                tf.matmul(self.last, self.weight) + self. bias)

            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores1, labels=self.input_y))

            self.predictions = tf.argmax(
                self.scores1, 1, name="predictions")
            dec = tf.cast(tf.argmax(self.input_y, 1) * self.predictions, tf.float32)
            self.concept_vect = tf.nn.embedding_lookup(
                self.W_concept, self.predictions)
            correct_predictions = tf.equal(
                self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_predictions, "float"), name="accuracy")


            # here we minimize the distance without considering the prediction i.e., reward_class
            self.cos_distance = tf.reduce_mean(tf.losses.cosine_distance(tf.nn.l2_normalize(
                self.concept_vect, 1), tf.nn.l2_normalize((self.em), 1), 1)*dec)
            self.loss_rationale = tf.reduce_mean(-(tf.log(self.sampled_prob+1e-8)*tf.expand_dims(dec,1) * self.lambda_val)

            self.cost = self.cross_entropy + self.cos_distance + self.loss_rationale
