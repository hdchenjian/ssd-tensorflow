import tensorflow as tf
import numpy as np


def classifier(x, size, mapsize, name):
    with tf.variable_scope(name):
        w = tf.get_variable("filter",
                            shape=[3, 3, x.get_shape()[3], size],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.zeros(size), name='biases')
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.reshape(x, [-1, mapsize.w*mapsize.h, size])
        l2 = tf.nn.l2_loss(w)
    return x, l2

def smooth_l1_loss(x):
    square_loss   = 0.5*x**2
    absolute_loss = tf.abs(x)
    return tf.where(tf.less(absolute_loss, 1.), square_loss, absolute_loss-0.5)

def array2tensor(x, name):
    init = tf.constant_initializer(value=x, dtype=tf.float32)
    tensor = tf.get_variable(name=name, initializer=init, shape=x.shape)
    return tensor

def l2_normalization(x, initial_scale, channels, name):
    with tf.variable_scope(name):
        scale = array2tensor(initial_scale*np.ones(channels), 'scale')
        x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x

class SSDVGG:
    def __init__(self, session, preset, _data_format='channels_last'):
        self.preset = preset
        self.session = session
        self.__built = False
        self.__build_names()
        if _data_format == 'channels_first': self.data_format = 'NCHW'
        else: self.data_format = 'NHWC'
        with tf.name_scope('define_input'):
            self.image_input = tf.placeholder(tf.float32, shape=(None, 300, 300, 3), name='image_input')
        print('self.new_scopes', self.new_scopes)
        print(self.preset.maps, len(self.preset.maps), '\n')

    def build_from_vgg(self, vgg_dir, num_classes):
        """
        Build the model for training based on a pre-define vgg16 model.
        :param num_classes:   number of classes
        """
        self.num_classes = num_classes+1
        self.num_vars = num_classes+5
        self.l2_loss = 0
        self.l2_weight_loss = []
        self.__load_vgg()
        self.norm_conv4_3 = l2_normalization(self._conv4_block, 20, 512, 'l2_norm_conv4_3')
        self.__maps = [self.norm_conv4_3, self._conv7, self._conv8_block,
                       self._conv9_block,  self._conv10_block,  self._conv11_block]
        print('l2_weight_loss', self.l2_weight_loss)
        for w in self.l2_weight_loss:
            self.l2_loss += tf.nn.l2_loss(w)
        self.__build_classifiers()
        self.__built = True


    def weight_variable(self, shape, name='name'):
        print('weight_variable', name);
        with tf.variable_scope(name):
            w = tf.get_variable("filter", shape=shape,
                                initializer=tf.contrib.layers.xavier_initializer())
            return w

    def bias_variable(self, shape, name='name'):
        print('bias_variable', name)
        with tf.variable_scope(name):
            return tf.Variable(tf.zeros(shape), name='biases')

    def conv_block(self, bottom, num_blocks, filters, kernel_size, pre_channel_size, strides, name, dilations=None):
        print('conv_block', name, '\n')
        with tf.variable_scope(name):
            _input = bottom
            for ind in range(1, num_blocks + 1):
                w = self.weight_variable([kernel_size, kernel_size, pre_channel_size, filters],
                                         name='{}_{}'.format(name, ind))
                self.l2_weight_loss.append(w)
                b = self.bias_variable(filters, name='{}_{}'.format(name, ind))
                print(_input, w, b, '\n')
                x = tf.nn.conv2d(
                    _input, w, strides, padding='SAME', data_format=self.data_format, dilations=dilations,
                    name='{}_{}'.format(name, ind))
                print(_input, w, b, x, '\n')
                conv = tf.nn.relu(tf.nn.bias_add(x, b))
                _input = conv
                pre_channel_size = filters
            return _input

    def __load_vgg(self):
        self._conv1_block = self.conv_block(self.image_input, 2, 64, 3, 3, [1, 1, 1, 1], 'conv1')
        self._pool1 = tf.nn.max_pool(self._conv1_block, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     data_format=self.data_format, name='pool1')
        self._conv2_block = self.conv_block(self._pool1, 2, 128, 3, 64, [1, 1, 1, 1], 'conv2')
        self._pool2 = tf.nn.max_pool(self._conv2_block, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     data_format=self.data_format, name='pool2')
        self._conv3_block = self.conv_block(self._pool2, 3, 256, 3, 128, [1, 1, 1, 1], 'conv3')
        self._pool3 = tf.nn.max_pool(self._conv3_block, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     data_format=self.data_format, name='pool3')
        self._conv4_block = self.conv_block(self._pool3, 3, 512, 3, 256, [1, 1, 1, 1], 'conv4')
        self._pool4 = tf.nn.max_pool(self._conv4_block, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                     data_format=self.data_format, name='pool4')
        self._conv5_block = self.conv_block(self._pool4, 3, 512, 3, 512, [1, 1, 1, 1], 'conv5')
        self._pool5 = tf.nn.max_pool(self._conv5_block, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME',
                                     data_format=self.data_format, name='pool5')
        self._conv6 = self.conv_block(self._pool5, 1, 1024, 3, 512, [1, 1, 1, 1], name='fc6', dilations=[1, 6, 6, 1])
        self._conv7 = self.conv_block(self._conv6, 1, 1024, 1, 1024, [1, 1, 1, 1], name='fc7')

        # SSD layers
        self._conv8_block = self.ssd_conv_block(self._conv7, 256, 1024, [1, 2, 2, 1], 'conv8')
        self._conv9_block = self.ssd_conv_block(self._conv8_block, 128, 512, [1, 2, 2, 1], 'conv9')
        self._conv10_block = self.ssd_conv_block(
            self._conv9_block, 128, 256, [1, 1, 1, 1], 'conv10', padding='VALID')
        self._conv11_block = self.ssd_conv_block(
            self._conv10_block, 128, 256, [1, 1, 1, 1], 'conv11', padding='VALID')

    def ssd_conv_block(self, bottom, filters, pre_channel_size, strides, name, padding='SAME'):
        print('ssd_conv_block', name)
        with tf.variable_scope(name):
            w = self.weight_variable([1, 1, pre_channel_size, filters], name='{}_{}'.format(name, 1))
            self.l2_weight_loss.append(w)
            b = self.bias_variable([filters], name='{}_{}'.format(name, 1))
            conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
                bottom, w, [1, 1, 1, 1], padding=padding, data_format=self.data_format,
                name='{}_{}'.format(name, 1)), b))

            pre_channel_size = filters
            w2 = self.weight_variable([3, 3, pre_channel_size, filters * 2], name='{}_{}'.format(name, 2))
            self.l2_weight_loss.append(w2)
            b2 = self.bias_variable([filters * 2], name='{}_{}'.format(name, 2))
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
                conv, w2, strides, padding=padding, data_format=self.data_format,
                name='{}_{}'.format(name, 2)), b2))
            return conv2


    def __with_loss(self, x, l2_loss):
        self.l2_loss += l2_loss
        return x

    def __build_classifiers(self):
        with tf.variable_scope('classifiers'):
            self.__classifiers = []
            for i in range(len(self.__maps)):
                fmap = self.__maps[i]
                map_size = self.preset.maps[i].size
                for j in range(2+len(self.preset.maps[i].aspect_ratios)):
                    name = 'classifier{}_{}'.format(i, j)
                    clsfier, l2 = classifier(fmap, self.num_vars, map_size, name)
                    self.__classifiers.append(self.__with_loss(clsfier, l2))

        with tf.variable_scope('output'):
            output = tf.concat(self.__classifiers, axis=1, name='output')
            self.logits = output[:,:,:self.num_classes]

        with tf.variable_scope('result'):
            self.classifier = tf.nn.softmax(self.logits)
            self.locator = output[:,:,self.num_classes:]
            self.result = tf.concat([self.classifier, self.locator],
                                        axis=-1, name='result')

    def build_optimizer(self, learning_rate=0.001, weight_decay=0.0005,
                        momentum=0.9, global_step=None):

        self.labels = tf.placeholder(tf.float32, name='labels',
                                    shape=[None, None, self.num_vars])

        with tf.variable_scope('ground_truth'):
            # Split the ground truth tensor
            # Classification ground truth tensor
            # Shape: (batch_size, num_anchors, num_classes)
            gt_cl = self.labels[:,:,:self.num_classes]

            # Localization ground truth tensor
            # Shape: (batch_size, num_anchors, 4)
            gt_loc = self.labels[:,:,self.num_classes:]

            # Batch size
            # Shape: scalar
            batch_size = tf.shape(gt_cl)[0]

        # Compute match counters
        with tf.variable_scope('match_counters'):
            # Number of anchors per sample
            # Shape: (batch_size)
            total_num = tf.ones([batch_size], dtype=tf.int64) * \
                        tf.to_int64(self.preset.num_anchors)

            # Number of negative (not-matched) anchors per sample, computed
            # by counting boxes of the background class in each sample.
            # Shape: (batch_size)
            negatives_num = tf.count_nonzero(gt_cl[:,:,-1], axis=1)

            # Number of positive (matched) anchors per sample
            # Shape: (batch_size)
            positives_num = total_num-negatives_num

            # Number of positives per sample that is division-safe
            # Shape: (batch_size)
            positives_num_safe = tf.where(tf.equal(positives_num, 0),
                                          tf.ones([batch_size])*10e-15,
                                          tf.to_float(positives_num))

        # Compute masks
        with tf.variable_scope('match_masks'):
            # Boolean tensor determining whether an anchor is a positive
            # Shape: (batch_size, num_anchors)
            positives_mask = tf.equal(gt_cl[:,:,-1], 0)

            # Boolean tensor determining whether an anchor is a negative
            # Shape: (batch_size, num_anchors)
            negatives_mask = tf.logical_not(positives_mask)

        # Compute the confidence loss
        with tf.variable_scope('confidence_loss'):
            # Cross-entropy tensor - all of the values are non-negative
            # Shape: (batch_size, num_anchors)
            ce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=gt_cl,
                                                            logits=self.logits)

            # Sum up the loss of all the positive anchors
            # Positives - the loss of negative anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positives = tf.where(positives_mask, ce, tf.zeros_like(ce))

            # Total loss of positive anchors
            # Shape: (batch_size)
            positives_sum = tf.reduce_sum(positives, axis=-1)

            # Figure out what the negative anchors with highest confidence loss
            # are
            # Negatives - the loss of positive anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            negatives = tf.where(negatives_mask, ce, tf.zeros_like(ce))

            # Top negatives - sorted confience loss with the highest one first
            # Shape: (batch_size, num_anchors)
            negatives_top = tf.nn.top_k(negatives, self.preset.num_anchors)[0]

            # Fugure out what the number of negatives we want to keep is
            # Maximum number of negatives to keep per sample - we keep at most
            # 3 times as many as we have positive anchors in the sample
            # Shape: (batch_size)
            negatives_num_max = tf.minimum(negatives_num, 3*positives_num)

            # Mask out superfluous negatives and compute the sum of the loss
            # Transposed vector of maximum negatives per sample
            # Shape (batch_size, 1)
            negatives_num_max_t = tf.expand_dims(negatives_num_max, 1)

            # Range tensor: [0, 1, 2, ..., num_anchors-1]
            # Shape: (num_anchors)
            rng = tf.range(0, self.preset.num_anchors, 1)

            # Row range, the same as above, but int64 and a row of a matrix
            # Shape: (1, num_anchors)
            range_row = tf.to_int64(tf.expand_dims(rng, 0))

            # Mask of maximum negatives - first `negative_num_max` elements
            # in corresponding row are `True`, the rest is false
            # Shape: (batch_size, num_anchors)
            negatives_max_mask = tf.less(range_row, negatives_num_max_t)

            # Max negatives - all the positives and superfluous negatives are
            # zeroed out.
            # Shape: (batch_size, num_anchors)
            negatives_max = tf.where(negatives_max_mask, negatives_top,
                                     tf.zeros_like(negatives_top))

            # Sum of max negatives for each sample
            # Shape: (batch_size)
            negatives_max_sum = tf.reduce_sum(negatives_max, axis=-1)

            # Compute the confidence loss for each element
            # Total confidence loss for each sample
            # Shape: (batch_size)
            confidence_loss = tf.add(positives_sum, negatives_max_sum)

            # Total confidence loss normalized by the number of positives
            # per sample
            # Shape: (batch_size)
            confidence_loss = tf.where(tf.equal(positives_num, 0),
                                       tf.zeros([batch_size]),
                                       tf.div(confidence_loss,
                                              positives_num_safe))

            # Mean confidence loss for the batch
            # Shape: scalar
            self.confidence_loss = tf.reduce_mean(confidence_loss,
                                                  name='confidence_loss')

        # Compute the localization loss
        with tf.variable_scope('localization_loss'):
            # Element-wise difference between the predicted localization loss
            # and the ground truth
            # Shape: (batch_size, num_anchors, 4)
            loc_diff = tf.subtract(self.locator, gt_loc)

            # Smooth L1 loss
            # Shape: (batch_size, num_anchors, 4)
            loc_loss = smooth_l1_loss(loc_diff)

            # Sum of localization losses for each anchor
            # Shape: (batch_size, num_anchors)
            loc_loss_sum = tf.reduce_sum(loc_loss, axis=-1)

            # Positive locs - the loss of negative anchors is zeroed out
            # Shape: (batch_size, num_anchors)
            positive_locs = tf.where(positives_mask, loc_loss_sum,
                                     tf.zeros_like(loc_loss_sum))

            # Total loss of positive anchors
            # Shape: (batch_size)
            localization_loss = tf.reduce_sum(positive_locs, axis=-1)

            # Total localization loss normalized by the number of positives
            # per sample
            # Shape: (batch_size)
            localization_loss = tf.where(tf.equal(positives_num, 0),
                                         tf.zeros([batch_size]),
                                         tf.div(localization_loss,
                                                positives_num_safe))

            # Mean localization loss for the batch
            # Shape: scalar
            self.localization_loss = tf.reduce_mean(localization_loss,
                                                    name='localization_loss')

        # Compute total loss
        with tf.variable_scope('total_loss'):
            # Sum of the localization and confidence loss
            # Shape: (batch_size)
            self.conf_and_loc_loss = tf.add(self.confidence_loss,
                                            self.localization_loss,
                                            name='sum_losses')

            # L2 loss
            # Shape: scalar
            self.l2_loss = tf.multiply(weight_decay, self.l2_loss,
                                       name='l2_loss')

            # Final loss
            # Shape: scalar
            self.loss = tf.add(self.conf_and_loc_loss, self.l2_loss,
                               name='loss')

        # Build the optimizer
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            optimizer = optimizer.minimize(self.loss, global_step=global_step,
                                           name='optimizer')

        # Store the tensors
        self.optimizer = optimizer
        self.losses = {
            'total': self.loss,
            'localization': self.localization_loss,
            'confidence': self.confidence_loss,
            'l2': self.l2_loss
        }

    def build_from_metagraph(self, metagraph_file, checkpoint_file):
        """
        Build the model for inference from a metagraph shapshot and weights
        checkpoint.
        """
        sess = self.session
        saver = tf.train.import_meta_graph(metagraph_file)
        saver.restore(sess, checkpoint_file)
        self.image_input = sess.graph.get_tensor_by_name('image_input:0')
        self.result      = sess.graph.get_tensor_by_name('result/result:0')

    def build_optimizer_from_metagraph(self):
        """
        Get the optimizer and the loss from metagraph
        """
        sess = self.session
        self.loss = sess.graph.get_tensor_by_name('total_loss/loss:0')
        self.localization_loss = sess.graph.get_tensor_by_name('localization_loss/localization_loss:0')
        self.confidence_loss = sess.graph.get_tensor_by_name('confidence_loss/confidence_loss:0')
        self.l2_loss = sess.graph.get_tensor_by_name('total_loss/l2_loss:0')
        self.optimizer = sess.graph.get_operation_by_name('optimizer/optimizer')
        self.labels = sess.graph.get_tensor_by_name('labels:0')

        self.losses = {
            'total': self.loss,
            'localization': self.localization_loss,
            'confidence': self.confidence_loss,
            'l2': self.l2_loss
        }

    def __build_names(self):
        # Names of the original and new scopes
        self.original_scopes = [
            'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
            'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2',
            'conv5_3', 'fc6_1', 'fc7_1'
        ]

        self.new_scopes = [
            'conv8_1', 'conv8_2', 'conv9_1', 'conv9_2', 'conv10_1', 'conv10_2', 'conv11_1', 'conv11_2'
        ]

        if len(self.preset.maps) == 7:
            self.new_scopes += ['conv12_1', 'conv12_2']

        for i in range(len(self.preset.maps)):
            for j in range(2+len(self.preset.maps[i].aspect_ratios)):
                self.new_scopes.append('classifiers/classifier{}_{}'.format(i, j))

    def build_summaries(self, restore):
        if restore:
            return self.session.graph.get_tensor_by_name('net_summaries/net_summaries:0')

        # Build the filter summaries
        names = self.original_scopes + self.new_scopes
        sess = self.session
        with tf.variable_scope('filter_summaries'):
            summaries = []
            for name in names:
                if 'classifiers' not in name:
                    tensor_name = name.split('_')[0] + '/' + name+'/filter:0'
                else:
                    tensor_name = name+'/filter:0'
                tensor = sess.graph.get_tensor_by_name(tensor_name)
                summary = tf.summary.histogram(name, tensor)
                summaries.append(summary)

        # Scale summary
        with tf.variable_scope('scale_summary'):
            tensor = sess.graph.get_tensor_by_name('l2_norm_conv4_3/scale:0')
            summary = tf.summary.histogram('l2_norm_conv4_3', tensor)
            summaries.append(summary)

        return tf.summary.merge(summaries, name='net_summaries')
