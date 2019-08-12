import numpy as np
import argparse
import glob
from collections import defaultdict
import sys
sys.path.insert(0, './')

from vmetrics.data_helper import readKITTI, read_results, IMAGENETVID_CLASSES
from vmetrics.utils import overlap

W = 10
lambd = 0.2
Penalty = -0.03
IOU_thresh = 0.5
conf = 0.6

def window_func(t, W, lambd):
    return (1 + np.exp(-lambd * W)) / (1 + np.exp(lambd * (t - W)))

def rank_by_conf(bboxes, scores, cls_inds):
    idx = np.argsort(scores)[::-1]
    return bboxes[idx], scores[idx], cls_inds[idx]

def calc_nab(bboxes_gt, scores_gt, cls_gt, track_ids, bboxes, scores, cls, conf):
    first_frame_pertrack = defaultdict(lambda: 1e9)
    pos_score = 0.0
    neg_score = 0.0

    frames = max(bboxes_gt.keys() + bboxes.keys())
    for frame in range(frames):
        for c, track_id in zip(cls_gt[frame], track_ids[frame]):
            first_frame_pertrack[track_id] = min(first_frame_pertrack[track_id], frame)

        bboxes_tmp, scores_tmp, cls_tmp = bboxes[frame], scores[frame], cls[frame]
        if len(scores_tmp) > 1:
            bboxes_tmp, scores_tmp, cls_tmp = rank_by_conf(bboxes_tmp, scores_tmp, cls_tmp)

        num_gt = len(bboxes_gt[frame])
        assigned = np.zeros(num_gt, dtype='i')
        for idx, (bb, s, c) in enumerate(zip(bboxes_tmp, scores_tmp, cls_tmp)):
            if s < conf:
                continue
            max_overl = 0.0
            track_id = -1
            match_id = -1
            for id_gt in range(num_gt):
                if assigned[id_gt] > 0:
                    continue
                overl = overlap(bboxes_gt[frame][id_gt], bb)
                c_gt = cls_gt[frame][id_gt]
                if overl > max_overl and c_gt == c:
                    max_overl = overl
                    track_id = track_ids[frame][id_gt]
                    match_id = id_gt
            if max_overl > IOU_thresh:
                assigned[match_id] = 1
                pos_score += window_func(frame - first_frame_pertrack[track_id], W, lambd)
            else:
                neg_score += 1
    return pos_score, neg_score

def eval_nab(res_files, gt_dir, conf):
    nab_pos_score = 0.0
    nab_neg_score = 0.0
    oracle_score = 0.0
    for res_file in res_files:
        tag = res_file.split('/')[-1].split('.')[0]
        gt_file = gt_dir + '/%s.txt'%tag
        bboxes_gt, scores_gt, cls_inds_gt, track_ids, occlusions, truncations = readKITTI(gt_file, with_extra=True, classes=IMAGENETVID_CLASSES)
        bboxes, scores, cls_inds = read_results(res_file)

        nab_pos, nab_neg = calc_nab(bboxes_gt, scores_gt, cls_inds_gt, track_ids, bboxes, scores, cls_inds, conf)
        orcale_pos, _ = calc_nab(bboxes_gt, scores_gt, cls_inds_gt, track_ids, bboxes_gt, scores_gt, cls_inds_gt, conf)
        oracle_score += orcale_pos
        nab_pos_score += nab_pos
        nab_neg_score += nab_neg
    nab_score = nab_pos_score + Penalty * nab_neg_score
    return nab_score / oracle_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('res_dir')
    parser.add_argument('gt_dir')
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()

    all_res_files = sorted(glob.glob(args.res_dir + '/*.txt'))
    gt_tags = glob.glob(args.gt_dir + '/*.txt')
    gt_tags = map(lambda x: x.split('/')[-1].split('.')[0], gt_tags)
    all_res_files = filter(lambda x: x.split('/')[-1].split('.')[0] in gt_tags, all_res_files)
    num_files = len(all_res_files)

    print "NAB Score at Conf %.2f:"%conf, eval_nab(all_res_files, args.gt_dir, conf=conf)
