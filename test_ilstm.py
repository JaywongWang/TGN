import os
import numpy as np
import json
import h5py
from opt import *
from data_provider import *
from model import *

import tensorflow as tf
import sys

# set default encoding
reload(sys)
sys.setdefaultencoding('utf-8')

np.set_printoptions(threshold='nan')


def getKey(item):
    return item['score']

""" Get tIoU of two segments
"""
def get_iou(pred, gt):
    start_pred, end_pred = pred
    start, end = gt
    intersection = max(0, min(end, end_pred) - max(start, start_pred))
    union = min(max(end, end_pred) - min(start, start_pred), end-start + end_pred-start_pred)
    iou = float(intersection) / (union + 1e-8)

    return iou

""" Non-Maximum Suppression
"""
def nms_detections(proposals, overlap=0.7):
    """Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously selected
    detection. This version is translated from Matlab code by Tomasz
    Malisiewicz, who sped up Pedro Felzenszwalb's code.

    Parameters
    ----------
    proposals: list of item, each item is a dict containing 'timestamp' and 'score' field

    Returns
    -------
    new proposals with only the proposals selected after non-maximum suppression.
    """
    if len(proposals) == 0:
        return proposals

    props = np.array([item['framestamp'] for item in proposals])
    scores = np.array([item['score'] for item in proposals])
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]

    out_proposals = []
    for idx in range(nms_props.shape[0]):
        prop = nms_props[idx].tolist()
        score = float(nms_scores[idx])
        out_proposals.append({'framestamp': prop, 'score': score})


    return out_proposals


""" Get R@k for all predictions
R@k: Given k proposals, if there is at least one proposal has higher tIoU than iou_threshold, R@k=1; otherwise R@k=0
The predictions should have been sorted by confidence
"""
def get_recall_at_k(predictions, groundtruths, iou_threshold=0.5, max_proposal_num=5):
    hit = np.zeros(shape=(len(groundtruths.keys()),), dtype=np.float32)

    for idd, idx in enumerate(groundtruths.keys()):
        if idx in predictions.keys():
            preds = predictions[idx][:max_proposal_num]
            for pred in preds:
                if get_iou(pred['framestamp'], groundtruths[idx]['framestamp']) >= iou_threshold:
                    hit[idd] = 1.

    avg_recall = np.sum(hit) / len(hit)
    return avg_recall


def test(options):
    
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth=True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(options['gpu_id'])[1:-1]
    sess = tf.InteractiveSession(config=sess_config)

    # build model
    print('Building model ...')
    model = TGN(options)
    inputs, outputs = model.build_inference()


    # print variable names
    for v in tf.trainable_variables():
        print(v.name)
        print(v.get_shape())

    print('Loading data ...')
    data_provision = DataProvision(options)

    print('Restoring model from %s'%options['init_from'])
    saver = tf.train.Saver()
    saver.restore(sess, options['init_from'])

    batch_size = 4083
    split = 'test'

    test_batch_generator = data_provision.iterate_batch(split, batch_size)
    unique_anno_ids = data_provision.get_ids(split)
    anchors = data_provision.get_anchors()
    grounding = data_provision.get_grounding(split)

    c3d_resolution = options['c3d_resolution']

    print('Start to predict ...')
    t0 = time.time()

    count = 0

    # output data, for evaluation
    out_data = {}
    out_data['results'] = {}
    results = {}

    for batch_data in test_batch_generator:
        video_feats = batch_data['video_feat']
        video_feat_mask = batch_data['video_feat_mask']
        max_feat_len = video_feat_mask.shape[-1]
        
        this_batch_size = video_feat_mask.shape[0]
        zero_state = np.zeros(shape=(this_batch_size, options['rnn_size']))
        
        video_c_state = video_h_state = zero_state
        interactor_c_state = interactor_h_state = zero_state

        proposal_scores = np.zeros(shape=(this_batch_size, 0, options['num_anchors']))
        print('max_feat_len: {}'.format(max_feat_len))

        for video_feat_id in range(max_feat_len):
            print('Loop: {}'.format(video_feat_id))
            video_feat = video_feats[:, video_feat_id]
            batch_data['video_feat'] = video_feat
            batch_data['video_c_state'] = video_c_state
            batch_data['video_h_state'] = video_h_state
            batch_data['interactor_c_state'] = interactor_c_state
            batch_data['interactor_h_state'] = interactor_h_state

            feed_dict = {}
            for key, value in batch_data.items():
                if key not in inputs:
                    continue
                feed_dict[inputs[key]] = value

            proposal_score, video_c_state, video_h_state, interactor_c_state, interactor_h_state = sess.run([outputs['proposal_score'], outputs['video_c_state'], outputs['video_h_state'], outputs['interactor_c_state'], outputs['interactor_h_state']], feed_dict=feed_dict)
            proposal_score = np.expand_dims(proposal_score, axis=1)
            proposal_scores = np.concatenate((proposal_scores, proposal_score), axis=1)
        
        feat_lens = np.sum(video_feat_mask, axis=-1)
        for sample_id in range(this_batch_size):
            unique_anno_id = unique_anno_ids[count]
            feat_len = feat_lens[sample_id]
            # small gap due to feature resolution
            gap = 8

            print('%d-th video-query: %s, feat_len: %d'%(count, unique_anno_id, feat_len))
            
            result = []
            for i in range(feat_len):
                for j in range(options['num_anchors']):
                    # calculate time stamp from feature id
                    end_frame_id = round((i+0.5)*c3d_resolution)
                    start_frame_id = end_frame_id - anchors[j] + 1
                    
                    if start_frame_id >= 0.-gap:
                        start_frame_id = max(0., start_frame_id)
                        result.append({'framestamp': [start_frame_id, end_frame_id], 'score': float(proposal_scores[sample_id, i, j])})
                            

            print('Number of proposals (before post-processing): %d'%len(result))

            result = sorted(result, key=getKey, reverse=True)

            # score thresholding
            #result = [item for item in result if item['score'] >= options['proposal_score_threshold']]
            #print('Number of proposals (after score threshold): %d'%len(result))

            # keep high score elements
            result = result[:options['proposal_num_bf_nms']]

            # non-maximum suppresion
            result = nms_detections(result, overlap=options['nms_threshold'])

            print('Number of proposals (after nms): %d'%len(result))

            
            print('#{}, {}'.format(count, unique_anno_id))
            print('sentence query:')
            sentence_query = grounding[unique_anno_id]['raw_sentence']
            print(sentence_query)
            print('result (top 10):')
            print(result[:10])
            print('groundtruth:')
            print(grounding[unique_anno_id]['framestamp'])

            results[unique_anno_id] = result

            count = count + 1


    out_data['results'] = results

    out_json_file = 'results/%d/predict_proposals_%s_nms_%.2f.json'%(options['train_id'], split, options['nms_threshold'])

    rootfolder = os.path.dirname(out_json_file)
    if not os.path.exists(rootfolder):
        os.makedirs(rootfolder)

    print('Writing result json file ...')
    with open(out_json_file, 'w') as fid:
        json.dump(out_data, fid)

    #ground_truth_filename = os.path.join(options['grounding_data_path'], '{}.json'.format(split))
    #ground_truths = json.load(open(ground_truth_filename, 'r'))

    print('Evaluating ...')
    recall_at_k = get_recall_at_k(results, grounding, options['tiou_measure'], options['max_proposal_num'])

    print('Recall at {}: {}'.format(options['max_proposal_num'], recall_at_k))
    
    print('Total running time: %f seconds.'%(time.time()-t0))


if __name__ == '__main__':

    '''
    cur_options = default_options()
    # load original options from given path
    status_file = os.path.join(os.path.dirname(cur_options['init_from']), 'status.json')
    if os.path.exists(status_file):
        status = json.load(open(status_file))
        options = status['options']
    options['beam_size'] = cur_options['beam_size']
    options['max_proposal_num'] = cur_options['max_proposal_num']
    options['gpu_id'] = cur_options['gpu_id']
    options['init_from'] = cur_options['init_from']
    options['proposal_score_threshold'] = cur_options['proposal_score_threshold']
    options['nms_threshold'] = cur_options['nms_threshold']
    
    for key in cur_options:
        if key not in options:
            options[key] = cur_options[key]
    '''

    options = default_options()

    test(options)
