from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from constants import constants


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def cosineLoss(A, B, name):
    ''' A, B : (BatchSize, d) '''
    dotprod = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(A,1), tf.nn.l2_normalize(B,1)), 1)
    loss = 1-tf.reduce_mean(dotprod, name=name)
    return loss


def flatten(x):
    return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


def conv2d(x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
    with tf.variable_scope(name):
        stride_shape = [1, stride[0], stride[1], 1]
        filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:3])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = np.prod(filter_shape[:2]) * num_filters
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        return tf.nn.conv2d(x, w, stride_shape, pad) + b


def deconv2d(x, out_shape, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None, prevNumFeat=None):
    with tf.variable_scope(name):
        num_filters = out_shape[-1]
        prevNumFeat = int(x.get_shape()[3]) if prevNumFeat is None else prevNumFeat
        stride_shape = [1, stride[0], stride[1], 1]
        # transpose_filter : [height, width, out_channels, in_channels]
        filter_shape = [filter_size[0], filter_size[1], num_filters, prevNumFeat]

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[:2]) * prevNumFeat
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = np.prod(filter_shape[:3])
        # initialize weights with random weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                            collections=collections)
        b = tf.get_variable("b", [num_filters], initializer=tf.constant_initializer(0.0),
                            collections=collections)
        deconv2d = tf.nn.conv2d_transpose(x, w, tf.pack(out_shape), stride_shape, pad)
        # deconv2d = tf.reshape(tf.nn.bias_add(deconv2d, b), deconv2d.get_shape())
        return deconv2d

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keep_dims=True), 1), [1])
    return tf.one_hot(value, d)


def inverseUniverseHead(x, final_shape, nConvs=4):
    ''' universe agent example
        input: [None, 288]; output: [None, 42, 42, 1];
    '''
    print('Using inverse-universe head design')
    bs = tf.shape(x)[0]
    deconv_shape1 = [final_shape[1]]
    deconv_shape2 = [final_shape[2]]
    for i in range(nConvs):
        deconv_shape1.append((deconv_shape1[-1]-1)/2 + 1)
        deconv_shape2.append((deconv_shape2[-1]-1)/2 + 1)
    inshapeprod = np.prod(x.get_shape().as_list()[1:]) / 32.0
    assert(inshapeprod == deconv_shape1[-1]*deconv_shape2[-1])
    # print('deconv_shape1: ',deconv_shape1)
    # print('deconv_shape2: ',deconv_shape2)

    x = tf.reshape(x, [-1, deconv_shape1[-1], deconv_shape2[-1], 32])
    deconv_shape1 = deconv_shape1[:-1]
    deconv_shape2 = deconv_shape2[:-1]
    for i in range(nConvs-1):
        x = tf.nn.elu(deconv2d(x, [bs, deconv_shape1[-1], deconv_shape2[-1], 32],
                        "dl{}".format(i + 1), [3, 3], [2, 2], prevNumFeat=32))
        deconv_shape1 = deconv_shape1[:-1]
        deconv_shape2 = deconv_shape2[:-1]
    x = deconv2d(x, [bs] + final_shape[1:], "dl4", [3, 3], [2, 2], prevNumFeat=32)
    return x


def universeHead(x, nConvs=4):
    ''' universe agent example
        input: [None, 42, 42, 1]; output: [None, 288];
    '''
    print('Using universe head design')
    for i in range(nConvs):
        x = tf.nn.elu(conv2d(x, 32, "l{}".format(i + 1), [3, 3], [2, 2]))
        # print('Loop{} '.format(i+1),tf.shape(x))
        # print('Loop{}'.format(i+1),x.get_shape())
    x = flatten(x)
    return x


def nipsHead(x):
    ''' DQN NIPS 2013 and A3C paper
        input: [None, 84, 84, 4]; output: [None, 2592] -> [None, 256];
    '''
    print('Using nips head design')
    x = tf.nn.relu(conv2d(x, 16, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 32, "l2", [4, 4], [2, 2], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x


def natureHead(x):
    ''' DQN Nature 2015 paper
        input: [None, 84, 84, 4]; output: [None, 3136] -> [None, 512];
    '''
    print('Using nature head design')
    x = tf.nn.relu(conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
    x = tf.nn.relu(conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
    x = flatten(x)
    x = tf.nn.relu(linear(x, 512, "fc", normalized_columns_initializer(0.01)))
    return x


def doomHead(x):
    ''' Learning by Prediction ICLR 2017 paper
        (their final output was 64 changed to 256 here)
        input: [None, 120, 160, 1]; output: [None, 1280] -> [None, 256];
    '''
    print('Using doom head design')
    x = tf.nn.elu(conv2d(x, 8, "l1", [5, 5], [4, 4]))
    x = tf.nn.elu(conv2d(x, 16, "l2", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 32, "l3", [3, 3], [2, 2]))
    x = tf.nn.elu(conv2d(x, 64, "l4", [3, 3], [2, 2]))
    x = flatten(x)
    x = tf.nn.elu(linear(x, 256, "fc", normalized_columns_initializer(0.01)))
    return x

def linear(x, size, name, initializer=None, bias_init=0):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
    b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
    return tf.matmul(x, w) + b

class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, designHead='universe', 
                 add_cur_model=False, add_con_model=False):
        self.x = x = tf.placeholder(tf.float32, [None] + list(ob_space), name='x')
        size = 256
        if designHead == 'nips':
            x = nipsHead(x)
        elif designHead == 'nature':
            x = natureHead(x)
        elif designHead == 'doom':
            x = doomHead(x)
        elif 'tile' in designHead:
            x = universeHead(x, nConvs=2)
        else:
            x = universeHead(x)

        if add_cur_model:
            with tf.variable_scope("cur_model"):
                def curiosity_model(x):
                    for i,size in enumerate(constants['CURIOSITY_SIZES']):
                        x = tf.nn.relu(linear(x, size, "cur_model_"+str(i), normalized_columns_initializer(0.01)))
                    return linear(x, ac_space, "cur_model_last", normalized_columns_initializer(0.01))
                self.curiosity_model = curiosity_model
                self.curiosity_predictions = curiosity_model(x)
                self.cur_model_sample = categorical_sample(self.curiosity_predictions, ac_space)[0, :]

        if add_con_model:
            with tf.variable_scope("con_model"):
                def consistency_model(x):
                    for i,size in enumerate(constants['CONSISTENCY_SIZES']):
                        x = tf.nn.relu(linear(x, size, "con_model_"+str(i), normalized_columns_initializer(0.01)))
                    return linear(x, ac_space, "con_model_last", normalized_columns_initializer(0.01))
                self.consistency_model = consistency_model
                self.consistency_predictions = consistency_model(x)
                self.con_model_sample = categorical_sample(self.consistency_predictions, ac_space)[0, :]

        # introduce a "fake" batch dimension of 1 to do LSTM over time dim
        x = tf.expand_dims(x, [0])
        lstm = rnn.rnn_cell.BasicLSTMCell(size, state_is_tuple=True)
        self.state_size = lstm.state_size
        step_size = tf.shape(self.x)[:1]

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]
        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c_in')
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.rnn_cell.LSTMStateTuple(c_in, h_in)
        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_c, lstm_h = lstm_state
        x = tf.reshape(lstm_outputs, [-1, size])
        
        if add_cur_model:
            x = tf.concat(concat_dim=1,values=[x, self.curiosity_predictions])
        if add_con_model:
            x = tf.concat(concat_dim=1, values=[x, self.consistency_predictions])
        
        self.vf = tf.reshape(linear(x, 1, "value", normalized_columns_initializer(1.0)), [-1])
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]

        # [0, :] means pick action of first state from batch. Hardcoded b/c
        # batch=1 during rollout collection. Its not used during batch training.
        self.logits = linear(x, ac_space, "action", normalized_columns_initializer(0.01))
        self.sample = categorical_sample(self.logits, ac_space)[0, :]
        self.probs = tf.nn.softmax(self.logits, dim=-1)[0, :]

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        # tf.add_to_collection('probs', self.probs)
        # tf.add_to_collection('sample', self.sample)
        # tf.add_to_collection('state_out_0', self.state_out[0])
        # tf.add_to_collection('state_out_1', self.state_out[1])
        # tf.add_to_collection('vf', self.vf)

    def get_initial_features(self):
        # Call this function to get reseted lstm memory cells
        return self.state_init

    def act(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})
    
    def act_from_1step_cur_model(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.cur_model_sample, {self.x: [ob]})

    def predict_curiosity(self, ob):
        sess = tf.get_default_session()
        return sess.run(self.curiosity_predictions, {self.x: [ob]})

    def act_inference(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run([self.probs, self.sample, self.vf] + self.state_out,
                        {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})

    def value(self, ob, c, h):
        sess = tf.get_default_session()
        return sess.run(self.vf, {self.x: [ob], self.state_in[0]: c, self.state_in[1]: h})[0]


class StateActionPredictor(object):
    def __init__(self, ob_space, ac_space, designHead='universe', imagined_weight=0.4,
                 no_stop_grads=False, stop_grads_forward=False, backward_model=False,
                 forward_sizes=[256], inverse_sizes=[256], activate_bug=False):
        self.ac_space = ac_space
        self.ob_space = ob_space

        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.imagined_weight = imagined_weight
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape, name="placeholder_s1")
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape, name="placeholder_s2")
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space], name="placeholder_asample")
        self.con_bonus_phi_2 = tf.placeholder(tf.float32, [None,None], name="placeholder_con_bonus")

        # feature encoding: phi1, phi2: [None, LEN]
        print('okay using an imagined weight of', imagined_weight)
        
        # settings that don't belong here
        output_size = 256
        batch_size = tf.shape(phi1)[0]
        num_imagined = batch_size

        if designHead == 'nips':
            phi1 = nipsHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = nipsHead(phi2)
        elif designHead == 'nature':
            phi1 = natureHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = natureHead(phi2)
        elif designHead == 'doom':
            phi1 = doomHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = doomHead(phi2)
        elif 'tile' in designHead:
            phi1 = universeHead(phi1, nConvs=2)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universeHead(phi2, nConvs=2)
        else:
            phi1 = universeHead(phi1)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                phi2 = universeHead(phi2)

        # forward model: f(phi1,asample) -> phi2
        # predict next feature embedding
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        def forward_model(phi1, asample):
            f = tf.concat(1, [phi1, asample])
            for i,size in enumerate(forward_sizes):
                f = tf.nn.relu(linear(f, size, "forward_"+str(i), normalized_columns_initializer(0.01)))
            return linear(f, phi1.get_shape()[1].value, "forward_last", normalized_columns_initializer(0.01))
        self.forward_model = forward_model
        self.guessed_phi2 = forward_model(phi1, asample)
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(self.guessed_phi2, phi2)), name='forwardloss')
        self.forwardloss = self.forwardloss * 288.0  # lenFeatures=288. Factored out to make hyperparams not depend on it.

        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        # predict action from feature embedding of s1 and s2
        def inverse_model(phi1, phi2):
            g = tf.concat(1,[phi1, phi2])
            for i,size in enumerate(inverse_sizes):
                g = tf.nn.relu(linear(g, size, "inverse_"+str(i), normalized_columns_initializer(0.01)))
            return linear(g, ac_space, "inverse_last", normalized_columns_initializer(0.01))
        self.inverse_model = inverse_model

        # compute inverse loss on real actions
        logits = inverse_model(phi1, phi2)
        self.ainvprobs = tf.nn.softmax(logits, dim=-1)
        aindex = tf.argmax(asample, axis=1)  # aindex: [batch_size,]
        self.invloss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits, aindex), name="invloss_real")

        # compute inverse loss on placeholder embedding
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            con_logits = inverse_model(phi1, self.con_bonus_phi_2)
        self.con_bonus = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        con_logits, aindex), name="invloss_real")
        
        # Imagine some actions and states that weren't encountered
        imagined_action_idxs = tf.random_uniform(dtype=tf.int32, minval=0, maxval=ac_space, shape=[num_imagined])
        imagined_actions = tf.one_hot(imagined_action_idxs, ac_space)
        imagined_start_states_idxs = tf.random_uniform(dtype=tf.int32, minval=0, maxval=batch_size, shape=[num_imagined])
        if no_stop_grads:
            print('Not stopping gradients from consistency to encoder')
            imagined_phi1 = tf.gather(phi1, imagined_start_states_idxs)
        else:
            print('Stopping gradients from consistency to encoder')
            imagined_phi1 = tf.stop_gradient(tf.gather(phi1, imagined_start_states_idxs), name="stop_gradient_consistency_to_encoder")

        # predict next state for imagined actions
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            if stop_grads_forward and not no_stop_grads:
                print('Stopping grads from consistency to forward model')
                imagined_phi2 = tf.stop_gradient(forward_model(imagined_phi1, imagined_actions), name="stop_grad_consistency_to_forward")
            else:
                print('Not stopping grads from consistency to forward model')
                imagined_phi2 = forward_model(imagined_phi1, imagined_actions)
            
        # compute inverse loss on imagined actions
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            imagined_logits = inverse_model(imagined_phi1, imagined_phi2)
        self.ainvprobs_imagined = tf.nn.softmax(imagined_logits, dim=-1)
        if activate_bug:
            self.invloss_imagined = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                   logits, imagined_action_idxs), name="invloss_imagined")
        else:
            self.invloss_imagined = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                    imagined_logits, imagined_action_idxs), name="invloss_imagined")

        # Compute aggregate inverses loss
        self.invloss = tf.add(self.invloss_real, imagined_weight * self.invloss_imagined, name="invloss")
        #(1.0 - imagined_weight) * 

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_act(self, s1, s2):
        '''
        returns action probability distribution predicted by inverse model
            input: s1,s2: [h, w, ch]
            output: ainvprobs: [ac_space]
        '''
        sess = tf.get_default_session()
        return sess.run(self.ainvprobs, {self.s1: [s1], self.s2: [s2]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        # error = sess.run([self.forwardloss, self.invloss],
        #     {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error[0], ' ErrorI:', error[1])
        error = sess.run(self.forwardloss,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        error = error * constants['PREDICTION_BETA']
        return error

    def consistency_pred_bonus(self, s1, asample):
        sess = tf.get_default_session()
        guessed_phi2 = sess.run(self.guessed_phi2, {self.s1: [s1], self.asample: [asample]})
        if len(np.shape(guessed_phi2)) > 2:
            guessed_phi2 = np.reshape(guessed_phi2, [1,-1])
        error = sess.run(self.con_bonus, {self.s1: [s1], self.con_bonus_phi_2: guessed_phi2,
                                             self.asample: [asample]})
        return error

    def consistency_bonus_all_actions(self, s1):
        actions = np.zeros((self.ac_space,self.ac_space))
        actions[np.arange(self.ac_space), np.arange(self.ac_space)] = 1.
        np.random.shuffle(actions)

        sess = tf.get_default_session()
        guessed_phi2 = sess.run(self.guessed_phi2, {self.s1: [s1], self.asample: actions})
        error = sess.run(self.con_bonus, {self.s1: [s1], self.con_bonus_phi_2: guessed_phi2,
                                             self.asample: [asample]})
        print("Size of consistency bonus error", np.shape(error))
        return error

class StatePredictor(object):
    '''
    Loss is normalized across spatial dimension (42x42), but not across batches.
    It is unlike ICM where no normalization is there across 288 spatial dimension
    and neither across batches.
    '''

    def __init__(self, ob_space, ac_space, designHead='universe', unsupType='state'):
        # input: s1,s2: : [None, h, w, ch] (usually ch=1 or 4)
        # asample: 1-hot encoding of sampled action from policy: [None, ac_space]
        input_shape = [None] + list(ob_space)
        self.s1 = phi1 = tf.placeholder(tf.float32, input_shape)
        self.s2 = phi2 = tf.placeholder(tf.float32, input_shape)
        self.asample = asample = tf.placeholder(tf.float32, [None, ac_space])
        self.stateAenc = unsupType == 'stateAenc'

        # feature encoding: phi1: [None, LEN]
        if designHead == 'universe':
            phi1 = universeHead(phi1)
            if self.stateAenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universeHead(phi2)
        elif 'tile' in designHead:  # for mario tiles
            phi1 = universeHead(phi1, nConvs=2)
            if self.stateAenc:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    phi2_aenc = universeHead(phi2)
        else:
            print('Only universe designHead implemented for state prediction baseline.')
            exit(1)

        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat(1, [phi1, asample])
        f = tf.nn.relu(linear(f, phi1.get_shape()[1].value, "f1", normalized_columns_initializer(0.01)))
        if 'tile' in designHead:
            f = inverseUniverseHead(f, input_shape, nConvs=2)
        else:
            f = inverseUniverseHead(f, input_shape)
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        if self.stateAenc:
            self.aencBonus = 0.5 * tf.reduce_mean(tf.square(tf.subtract(phi1, phi2_aenc)), name='aencBonus')
        self.predstate = phi1

        # variable list
        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

    def pred_state(self, s1, asample):
        '''
        returns state predicted by forward model
            input: s1: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: s2: [h, w, ch]
        '''
        sess = tf.get_default_session()
        return sess.run(self.predstate, {self.s1: [s1],
                                            self.asample: [asample]})[0, :]

    def pred_bonus(self, s1, s2, asample):
        '''
        returns bonus predicted by forward model
            input: s1,s2: [h, w, ch], asample: [ac_space] 1-hot encoding
            output: scalar bonus
        '''
        sess = tf.get_default_session()
        bonus = self.aencBonus if self.stateAenc else self.forwardloss
        error = sess.run(bonus,
            {self.s1: [s1], self.s2: [s2], self.asample: [asample]})
        # print('ErrorF: ', error)
        error = error * constants['PREDICTION_BETA']
        return error
