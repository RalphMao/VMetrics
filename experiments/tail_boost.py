import numpy as np
import argparse
import glob
from collections import defaultdict
import sys
sys.path.insert(0, './')

from vmetrics.data_helper import readKITTI, read_results, write_results, IMAGENETVID_CLASSES
from vmetrics.utils import overlap, filter_byconf, rcnn2gt, rcnn2pred, rcnn2apred
from vmetrics.ap import eval_mAP, eval_aAP


IOU_thresh = 0.5
def boost_frames(bboxes_gt, scores_gt, cls_gt, track_ids, bboxes, scores, cls, conf, lower_bound=True):
    first_frame_pertrack = defaultdict(lambda: 1e9)
    first_detect_pertrack = defaultdict(lambda: 1e9)
    class_pertrack = defaultdict(int)

    frames = max(bboxes_gt.keys() + bboxes.keys())
    results = [0] * 6
    num_boosted = 0

    for frame in range(frames):
        for c, track_id in zip(cls_gt[frame], track_ids[frame]):
            first_frame_pertrack[track_id] = min(first_frame_pertrack[track_id], frame)
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
            if max_overl > IOU_thresh:
                if first_detect_pertrack[track_id] > frame:
                    first_detect_pertrack[track_id] = frame
                if (frame - first_detect_pertrack[track_id] >= 10 or frame - first_frame_pertrack[track_id] >= 30) and scores[frame][idx] < 1.0:
                    scores[frame][idx] = 1.0
                    num_boosted += 1
    return num_boosted

def boost_at_conf(res_files, gt_dir, new_dir, conf):
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    total_num_boosted = 0
    for res_file in res_files:
        tag = res_file.split('/')[-1].split('.')[0]
        gt_file = gt_dir + '/%s.txt'%tag
        new_res = new_dir + '/%s.txt'%tag
        bboxes_gt, scores_gt, cls_inds_gt, track_ids, occlusions, truncations = readKITTI(gt_file, with_extra=True, classes=IMAGENETVID_CLASSES)
        bboxes, scores, cls_inds = read_results(res_file)

        total_num_boosted += boost_frames(bboxes_gt, scores_gt, cls_inds_gt, track_ids, bboxes, scores, cls_inds, conf)
        write_results(new_res, bboxes, scores, cls_inds)
    print "Boost %d objects"%total_num_boosted

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
        boost_at_conf(res_files, args.gt_dir, args.new_res_dir, 0.2)
