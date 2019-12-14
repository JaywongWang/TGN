# -*- coding: utf-8 -*-

'''
Model Implementation
'''

import tensorflow as tf
import math


class TGN(object):

    def __init__(self, options):
        self.options = options

    def build_inference(self, reuse=False):
        """
        Build inference model for generating next states
        """

        inputs = {}
        outputs = {}

        video_feat = tf.placeholder(tf.float32, [None, self.options['video_feat_dim']], name='video_feat')
        sentence = tf.placeholder(tf.float32, [None, self.options['max_sentence_len'], self.options['word_embed_size']])
        sentence_mask = tf.placeholder(tf.float32, [None, None])

        if self.options['bidirectional_lstm_sentence']:
            sentence_bw = tf.placeholder(tf.float32,
                                         [None, self.options['max_sentence_len'], self.options['word_embed_size']])
            inputs['sentence_bw'] = sentence_bw

        video_c_state = tf.placeholder(tf.float32, [None, self.options['rnn_size']])
        video_h_state = tf.placeholder(tf.float32, [None, self.options['rnn_size']])

        interactor_c_state = tf.placeholder(tf.float32, [None, self.options['rnn_size']])
        interactor_h_state = tf.placeholder(tf.float32, [None, self.options['rnn_size']])

        inputs['video_feat'] = video_feat
        inputs['sentence'] = sentence
        inputs['sentence_mask'] = sentence_mask
        inputs['video_c_state'] = video_c_state
        inputs['video_h_state'] = video_h_state
        inputs['interactor_c_state'] = interactor_c_state
        inputs['interactor_h_state'] = interactor_h_state

        video_state = tf.nn.rnn_cell.LSTMStateTuple(video_c_state, video_h_state)
        interactor_state = tf.nn.rnn_cell.LSTMStateTuple(interactor_c_state, interactor_h_state)

        batch_size = tf.shape(video_feat)[0]

        rnn_cell_sentence = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_video = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_interator = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )

        with tf.variable_scope('sentence_encoding', reuse=reuse) as sentence_scope:
            #sequence_length = tf.fill([batch_size, ], self.options['max_sentence_len'])
            sequence_length = tf.reduce_sum(sentence_mask, axis=-1)
            initial_state = rnn_cell_sentence.zero_state(batch_size=batch_size, dtype=tf.float32)

            sentence_states, sentence_final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell_sentence,
                inputs=sentence,
                sequence_length=sequence_length,
                initial_state=initial_state,
                dtype=tf.float32
            )

            if self.options['bidirectional_lstm_sentence']:
                rnn_cell_sentence_bw = tf.contrib.rnn.LSTMCell(
                    num_units=self.options['rnn_size'],
                    state_is_tuple=True,
                    initializer=tf.orthogonal_initializer()
                )
                with tf.variable_scope('sentence_bw') as scope:
                    sentence_states_bw, sentence_final_state_bw = tf.nn.dynamic_rnn(
                        cell=rnn_cell_sentence_bw,
                        inputs=sentence_bw,
                        sequence_length=sequence_length,
                        initial_state=initial_state,
                        dtype=tf.float32
                    )
                    sentence_states_bw = tf.reverse_sequence(sentence_states_bw,
                                                             seq_lengths=tf.to_int32(sequence_length), seq_axis=1)
                sentence_states = tf.concat([sentence_states, sentence_states_bw], axis=-1)

        with tf.variable_scope('interactor', reuse=reuse) as interactor_scope:
            sentence_states_reshape = tf.reshape(sentence_states, [-1, (
                        1 + int(self.options['bidirectional_lstm_sentence'])) * self.options['rnn_size']])

            # get video state
            with tf.variable_scope('video_rnn') as video_rnn_scope:
                _, video_state = rnn_cell_video(inputs=video_feat, state=video_state)

            video_c_state, video_h_state = video_state

            # calculate attention over words
            # use a one-layer network to do this
            with tf.variable_scope('word_attention', reuse=reuse) as attention_scope:
                h_states = tf.tile(tf.concat([interactor_h_state, video_h_state], axis=-1),
                                   [1, self.options['max_sentence_len']])
                h_states = tf.reshape(h_states, [-1, 2 * self.options['rnn_size']])

                attention_input = tf.concat([h_states, sentence_states_reshape], axis=-1)

                attention_layer1 = tf.contrib.layers.fully_connected(
                    inputs=attention_input,
                    num_outputs=self.options['attention_hidden_size'],
                    activation_fn=tf.nn.tanh,
                    weights_initializer=tf.contrib.layers.xavier_initializer()
                )
                attention_layer2 = tf.contrib.layers.fully_connected(
                    inputs=attention_layer1,
                    num_outputs=1,
                    activation_fn=None,
                    weights_initializer=tf.contrib.layers.xavier_initializer()
                )

            # reshape to match
            attention_reshape = tf.reshape(attention_layer2, [-1, self.options['max_sentence_len']])
            attention_score = tf.nn.softmax(attention_reshape, dim=-1)
            attention_score = tf.reshape(attention_score, [-1, 1, self.options['max_sentence_len']])

            # attended word feature
            attended_word_feature = tf.matmul(attention_score,
                                              sentence_states)  # already support batch matrix multiplication in v1.0
            attended_word_feature = tf.reshape(attended_word_feature, [-1, (
                        1 + int(self.options['bidirectional_lstm_sentence'])) * self.options['rnn_size']])

            # calculate next interator state
            interactor_input = tf.concat([video_h_state, attended_word_feature], axis=-1)

            with tf.variable_scope('interactor_rnn') as interactor_rnn_scope:
                _, interactor_state = rnn_cell_interator(inputs=interactor_input, state=interactor_state)
            interactor_c_state, interactor_h_state = interactor_state

            with tf.variable_scope('predict_proposal'):
                logit_output = tf.contrib.layers.fully_connected(
                    inputs=interactor_h_state,
                    num_outputs=self.options['num_anchors'],
                    activation_fn=None
                )

                # score
                proposal_score = tf.sigmoid(logit_output, name='proposal_scores')

        outputs['proposal_score'] = proposal_score
        outputs['video_c_state'] = video_c_state
        outputs['video_h_state'] = video_h_state
        outputs['interactor_c_state'] = interactor_c_state
        outputs['interactor_h_state'] = interactor_h_state

        return inputs, outputs

    def build_train(self):
        """
        Build training model
        """

        inputs = {}
        outputs = {}

        video_feat = tf.placeholder(tf.float32, [None, None, self.options['video_feat_dim']], name='video_feat')
        video_feat_mask = tf.placeholder(tf.float32, [None, None])
        anchor_mask = tf.placeholder(tf.float32, [None, None, self.options['num_anchors']])
        sentence = tf.placeholder(tf.float32, [None, None, self.options['word_embed_size']])
        sentence_mask = tf.placeholder(tf.float32, [None, None])

        if self.options['bidirectional_lstm_sentence']:
            sentence_bw = tf.placeholder(tf.float32,
                                         [None, self.options['max_sentence_len'], self.options['word_embed_size']])
            inputs['sentence_bw'] = sentence_bw

        inputs['video_feat'] = video_feat
        inputs['video_feat_mask'] = video_feat_mask
        inputs['anchor_mask'] = anchor_mask
        inputs['sentence'] = sentence
        inputs['sentence_mask'] = sentence_mask

        ## proposal, densely annotated
        proposal = tf.placeholder(tf.int32, [None, None, self.options['num_anchors']], name='proposal')
        inputs['proposal'] = proposal

        ## weighting for positive/negative labels (solve imblance data problem)
        proposal_weight = tf.placeholder(tf.float32, [self.options['num_anchors'], 2], name='proposal_weight')
        inputs['proposal_weight'] = proposal_weight

        # fc dropout
        dropout = tf.placeholder(tf.float32)
        inputs['dropout'] = dropout

        # get batch size, which is a scalar tensor
        batch_size = tf.shape(video_feat)[0]

        rnn_cell_sentence = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_video = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )
        rnn_cell_interator = tf.contrib.rnn.LSTMCell(
            num_units=self.options['rnn_size'],
            state_is_tuple=True,
            initializer=tf.orthogonal_initializer()
        )

        rnn_cell_sentence = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_sentence,
            input_keep_prob=1.0 - dropout,
            output_keep_prob=1.0 - dropout
        )
        rnn_cell_video = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_video,
            input_keep_prob=1.0 - dropout,
            output_keep_prob=1.0 - dropout
        )
        rnn_cell_interator = tf.contrib.rnn.DropoutWrapper(
            rnn_cell_interator,
            input_keep_prob=1.0 - dropout,
            output_keep_prob=1.0 - dropout
        )

        with tf.variable_scope('sentence_encoding') as sentence_scope:
            #sequence_length = tf.fill([batch_size, ], self.options['max_sentence_len'])
            sequence_length = tf.reduce_sum(sentence_mask, axis=-1)
            initial_state = rnn_cell_sentence.zero_state(batch_size=batch_size, dtype=tf.float32)

            sentence_states, sentence_final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell_sentence,
                inputs=sentence,
                sequence_length=sequence_length,
                initial_state=initial_state,
                dtype=tf.float32
            )

            if self.options['bidirectional_lstm_sentence']:
                rnn_cell_sentence_bw = tf.contrib.rnn.LSTMCell(
                    num_units=self.options['rnn_size'],
                    state_is_tuple=True,
                    initializer=tf.orthogonal_initializer()
                )
                with tf.variable_scope('sentence_bw') as scope:
                    sentence_states_bw, sentence_final_state_bw = tf.nn.dynamic_rnn(
                        cell=rnn_cell_sentence_bw,
                        inputs=sentence_bw,
                        sequence_length=sequence_length,
                        initial_state=initial_state,
                        dtype=tf.float32
                    )
                    sentence_states_bw = tf.reverse_sequence(sentence_states_bw,
                                                             seq_lengths=tf.to_int32(sequence_length), seq_axis=1)
                sentence_states = tf.concat([sentence_states, sentence_states_bw], axis=-1)

        logit_outputs = tf.fill([batch_size, 0, self.options['num_anchors']], 0.)

        with tf.variable_scope('interactor') as interactor_scope:
            interactor_state = rnn_cell_interator.zero_state(batch_size=batch_size, dtype=tf.float32)
            video_state = rnn_cell_video.zero_state(batch_size=batch_size, dtype=tf.float32)
            sentence_states_reshape = tf.reshape(sentence_states, [-1, (
                        1 + int(self.options['bidirectional_lstm_sentence'])) * self.options['rnn_size']])
            for i in range(self.options['sample_len']):
                if i > 0:
                    interactor_scope.reuse_variables()

                # get video state
                with tf.variable_scope('video_rnn') as video_rnn_scope:
                    _, video_state = rnn_cell_video(inputs=video_feat[:, i, :], state=video_state)

                # calculate attention over words
                # use a one-layer network to do this
                with tf.variable_scope('word_attention') as attention_scope:
                    h_states = tf.tile(tf.concat([interactor_state[1], video_state[1]], axis=-1),
                                       [1, self.options['max_sentence_len']])
                    h_states = tf.reshape(h_states, [-1, 2 * self.options['rnn_size']])

                    attention_input = tf.concat([h_states, sentence_states_reshape], axis=-1)

                    attention_layer1 = tf.contrib.layers.fully_connected(
                        inputs=attention_input,
                        num_outputs=self.options['attention_hidden_size'],
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer()
                    )
                    attention_layer2 = tf.contrib.layers.fully_connected(
                        inputs=attention_layer1,
                        num_outputs=1,
                        activation_fn=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer()
                    )

                # reshape to match
                attention_reshape = tf.reshape(attention_layer2, [-1, self.options['max_sentence_len']])
                attention_score = tf.nn.softmax(attention_reshape, axis=-1)
                attention_score = tf.reshape(attention_score, [-1, 1, self.options['max_sentence_len']])

                # attended word feature
                attended_word_feature = tf.matmul(attention_score, sentence_states)
                attended_word_feature = tf.reshape(attended_word_feature, [-1, (
                            1 + int(self.options['bidirectional_lstm_sentence'])) * self.options['rnn_size']])

                # calculate next interator state
                interactor_input = tf.concat([video_state[1], attended_word_feature], axis=-1)

                with tf.variable_scope('interactor_rnn') as interactor_rnn_scope:
                    _, interactor_state = rnn_cell_interator(inputs=interactor_input, state=interactor_state)

                with tf.variable_scope('predict_proposal') as proposal_scope:
                    logit_output = tf.contrib.layers.fully_connected(
                        inputs=interactor_state[1],
                        num_outputs=self.options['num_anchors'],
                        activation_fn=None
                    )
                    logit_output = tf.expand_dims(logit_output, axis=1)
                    logit_outputs = tf.concat([logit_outputs, logit_output], axis=1)

        logit_outputs = tf.reshape(logit_outputs, [-1, self.options['num_anchors']])

        # weighting positive samples
        proposal_weight0 = tf.reshape(proposal_weight[:, 0], [-1, self.options['num_anchors']])
        # weighting negative samples
        proposal_weight1 = tf.reshape(proposal_weight[:, 1], [-1, self.options['num_anchors']])

        # tile
        proposal_weight0 = tf.tile(proposal_weight0, [tf.shape(logit_outputs)[0], 1])
        proposal_weight1 = tf.tile(proposal_weight1, [tf.shape(logit_outputs)[0], 1])

        # get weighted sigmoid xentropy loss
        # use tensorflow built-in function
        # weight1 will be always 1.
        proposal = tf.reshape(proposal, [-1, self.options['num_anchors']])
        proposal_loss_term = tf.nn.weighted_cross_entropy_with_logits(
            targets=tf.to_float(proposal), logits=logit_outputs, pos_weight=proposal_weight0)

        if self.options['anchor_mask']:
            proposal_loss_term = tf.reshape(anchor_mask, [-1, self.options['num_anchors']]) * proposal_loss_term

        proposal_loss_term = tf.reduce_sum(proposal_loss_term, axis=-1)
        proposal_loss_term = tf.reshape(proposal_loss_term, [-1])

        video_feat_mask = tf.reshape(video_feat_mask, [-1])
        proposal_loss = tf.reduce_sum((video_feat_mask * proposal_loss_term)) / tf.to_float(
            tf.reduce_sum(video_feat_mask))

        # summary data, for visualization using Tensorboard
        tf.summary.scalar('proposal_loss', proposal_loss)

        # outputs from proposal module
        outputs['loss'] = proposal_loss

        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        outputs['reg_loss'] = reg_loss

        return inputs, outputs
