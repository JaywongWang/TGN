import os
import json
import time
import numpy as np
from opt import default_options
from data_provider import DataProvision
from model import CBP
from util import evaluation_metric_util, mkdirs
import argparse
import tensorflow as tf
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = default_options()
    for key, value in options.items():
        parser.add_argument('--%s' % key, dest=key, type=type(value), default=None)
    args = parser.parse_args()
    args = vars(args)
    for key, value in args.items():
        if value is not None:
            options[key] = value

    options['ckpt_prefix'] = './checkpoints/3/'
    options['status_file'] = options['ckpt_prefix'] + 'status.json'

    mkdirs(options['ckpt_prefix'])

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    sess = tf.InteractiveSession(config=sess_config)
    # build model
    print('Building model ...')
    model = CBP(options)
    inputs, outputs = model.build_inference()
    interactor_inputs, interactor_outputs = model.build_interactor_self_attention_inference()
    proposal_inputs, proposal_outputs = model.build_proposal_prediction_inference()

    print('Loading data ...')
    data_provision = DataProvision(options)

    saver = tf.train.Saver()

    all_ckpt_files = sorted(glob.glob('checkpoints/3/*.ckpt'))
    for ckpt_file in all_ckpt_files:

        print('Restoring model from %s' % ckpt_file)
        saver.restore(sess, ckpt_file)

        split = 'test'
        print('Start to predict ...')
        t0 = time.time()

        out_data, recall_at_k = evaluation_metric_util(
            options, data_provision, sess, inputs, outputs,
            interactor_inputs=interactor_inputs, interactor_outputs=interactor_outputs,
            proposal_inputs=proposal_inputs, proposal_outputs=proposal_outputs, split=split)

        print('Cost time: {} seconds.'.format(time.time()-t0))

        # rename ckpt
        ckpt_name = os.path.basename(ckpt_file)
        rec_name = ckpt_name.split('_')[1]
        new_ckpt_file = ckpt_file.replace(rec_name, 'rec%.2f' % (100*recall_at_k))
        print('Renaming ...')
        os.system('mv {} {}'.format(ckpt_file, new_ckpt_file))
        os.system('mv {} {}'.format(ckpt_file + '.meta', new_ckpt_file + '.meta'))
