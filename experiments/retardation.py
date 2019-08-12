import numpy as np
import argparse
import glob
from collections import defaultdict
import sys
sys.path.insert(0, './')

from vmetrics.data_helper import readKITTI, read_results, write_results, IMAGENETVID_CLASSES
from vmetrics.utils import overlap, filter_byconf, rcnn2gt, rcnn2pred, rcnn2apred
from vmetrics.ap import eval_mAP, eval_aAP

def update_dict_with_prefix(dict1, dict2, prefix):
    for key in dict2:
        dict1[prefix + str(key)] = dict2[key]

def merge_cls(cls_inds, merge_table):
    for key in cls_inds:
        cls_inds[key] = merge_table[cls_inds[key]]

IOU_thresh = 0.5
def delay_frames(bboxes_gt, scores_gt, cls_gt, track_ids, bboxes, scores, cls, conf, lower_bound=True):
    first_frame_pertrack = defaultdict(lambda: 1e9)
    first_detect_pertrack = defaultdict(lambda: 1e9)
    last_frame_pertrack = defaultdict(int)
    class_pertrack = defaultdict(int)

    frames = max(bboxes_gt.keys() + bboxes.keys())
    results = [0] * 6
    num_delayed = 0
    for frame in range(frames):
        for c, track_id in zip(cls_gt[frame], track_ids[frame]):
            first_frame_pertrack[track_id] = min(first_frame_pertrack[track_id], frame)
            last_frame_pertrack[track_id] = max(first_frame_pertrack[track_id], frame)
            if class_pertrack[track_id] == 0:
                class_pertrack[track_id] = c
            else:
                assert class_pertrack[track_id] == c, \
                    "%d vs %d"%(class_pertrack[track_id], c)

        num_gt = len(bboxes_gt[frame])
        for idx, (bb, s, c) in enumerate(zip(bboxes[frame], scores[frame], cls[frame])):
            if lower_bound:
                if s < conf:
                    continue
            else:
                if s > conf:
                    continue
            max_overl = 0.0
            track_id = -1
            for id_gt in range(num_gt):
                   
                overl = overlap(bboxes_gt[frame][id_gt], bb)
                c_gt = cls_gt[frame][id_gt]
                if overl > max_overl and c_gt == c:
                    max_overl = overl
                    track_id = track_ids[frame][id_gt]
            if max_overl > IOU_thresh and first_detect_pertrack[track_id] > frame and scores[frame][idx] > 0:
                first_detect_pertrack[track_id] = frame
                scores[frame][idx] = 0.0
                num_delayed += 1
    return num_delayed

def delay_at_conf(res_files, gt_dir, new_dir, conf, lower_bound=True):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    total_num_delayed = 0
    for res_file in res_files:
        tag = res_file.split('/')[-1].split('.')[0]
        gt_file = gt_dir + '/%s.txt'%tag
        new_res = new_dir + '/%s.txt'%tag
        bboxes_gt, scores_gt, cls_inds_gt, track_ids, occlusions, truncations = readKITTI(gt_file, with_extra=True, classes=IMAGENETVID_CLASSES)
        bboxes, scores, cls_inds = read_results(res_file)

        total_num_delayed += delay_frames(bboxes_gt, scores_gt, cls_inds_gt, track_ids, bboxes, scores, cls_inds, conf, lower_bound=lower_bound)
        write_results(new_res, bboxes, scores, cls_inds)
    print "Delay %d objects"%total_num_delayed

def get_mAP(res_files, gt_dir):
    bboxes_gt = {}
    scores_gt = {}
    cls_inds_gt = {}
    bboxes = {}
    scores = {}
    cls_inds = {}
    for res_file in res_files:
        tag = res_file.split('/')[-1].split('.')[0]
        gt_file = gt_dir + '/%s.txt'%tag
        bboxes_gt_tmp, scores_gt_tmp, cls_inds_gt_tmp, track_ids, occlusions, truncations = readKITTI(gt_file, with_extra=True, classes=IMAGENETVID_CLASSES)
        bboxes_tmp, scores_tmp, cls_inds_tmp = read_results(res_file)

        update_dict_with_prefix(bboxes_gt, bboxes_gt_tmp, tag)
        update_dict_with_prefix(scores_gt, scores_gt_tmp, tag)
        update_dict_with_prefix(cls_inds_gt, cls_inds_gt_tmp, tag)

        update_dict_with_prefix(bboxes, bboxes_tmp, tag)
        update_dict_with_prefix(scores, scores_tmp, tag)
        update_dict_with_prefix(cls_inds, cls_inds_tmp, tag)

    groundtruths = rcnn2gt(bboxes_gt, scores_gt, cls_inds_gt)
    predictions = rcnn2pred(bboxes, scores, cls_inds)
    predictions_a = rcnn2apred(bboxes, scores, cls_inds)
    mAP, ap_dict = eval_mAP(groundtruths, predictions)
    ap, rec, prec, fp_rate = eval_aAP(groundtruths, predictions_a)
    target_precs = [0.9]
    target_fps = [1.0]
    confs = []
    confs2 = []
    for target_prec in target_precs:
        idx = np.where(np.array(prec) < target_prec)[0][0]
        conf = predictions_a[idx]['confidence']
        confs.append(conf)
    for target_fp in target_fps:
        idx = np.where(np.array(fp_rate) > target_fp)[0][0]
        conf = predictions_a[idx]['confidence']
        confs2.append(conf)

    return mAP, confs, confs2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('res_dir')
    parser.add_argument('gt_dir')
    parser.add_argument('new_res_dir')
    parser.add_argument('--times', default=1, type=int)
    args = parser.parse_args()

    for i in range(args.times):
        if i == 0:
            res_dir = args.res_dir
        else:
            res_dir = args.new_res_dir
        res_files = sorted(glob.glob(res_dir + '/*.txt'))
        mAP, confs, _ = get_mAP(res_files, args.gt_dir)
        print "mAP:", mAP
        delay_at_conf(res_files, args.gt_dir, args.new_res_dir, 1.01, lower_bound=False)
