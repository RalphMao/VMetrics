import numpy as np
import argparse
import glob
from collections import defaultdict

from aplib.kitti_helper import readKITTI, read_results
from aplib.utils import overlap, filter_byconf
from aplib import rcnn2gt, rcnn2pred, eval_mAP, eval_aAP, rcnn2apred

IMAGENETVID_CLASSES = ('__background__', 'airplane', 'antelope', 'bear', 'bicycle',
           'bird', 'bus', 'car', 'cattle',
           'dog', 'domestic_cat', 'elephant', 'fox',
           'giant_panda', 'hamster', 'horse', 'lion',
           'lizard', 'monkey', 'motorcycle', 'rabbit',
           'red_panda', 'sheep', 'snake', 'squirrel',
           'tiger', 'train', 'turtle', 'watercraft',
           'whale', 'zebra')

def update_dict_with_prefix(dict1, dict2, prefix):
    for key in dict2:
        dict1[prefix + str(key)] = dict2[key]

def merge_cls(cls_inds, merge_table):
    for key in cls_inds:
        cls_inds[key] = merge_table[cls_inds[key]]

def calc_delay(bboxes_gt, scores_gt, cls_gt, track_ids, bboxes, scores, cls):
    first_frame_pertrack = defaultdict(lambda: 1e9)
    first_detect_pertrack = defaultdict(lambda: 1e9)
    last_frame_pertrack = defaultdict(int)
    class_pertrack = defaultdict(int)
    size_pertrack = defaultdict(list)

    frames = max(bboxes_gt.keys() + bboxes.keys())
    results = [0] * 6
    IOU_thresh = 0.5
    for frame in range(frames):
        for c, track_id, bbox in zip(cls_gt[frame], track_ids[frame], bboxes_gt[frame]):
            h, w = (bbox[3] - bbox[1], bbox[2] - bbox[0])
            shorter = min(h, w)
            size_pertrack[track_id].append(shorter)
            first_frame_pertrack[track_id] = min(first_frame_pertrack[track_id], frame)
            last_frame_pertrack[track_id] = max(first_frame_pertrack[track_id], frame)
            if class_pertrack[track_id] == 0:
                class_pertrack[track_id] = c
            else:
                assert class_pertrack[track_id] == c, \
                    "%d vs %d"%(class_pertrack[track_id], c)

        num_gt = len(bboxes_gt[frame])
        for bb, s, c in zip(bboxes[frame], scores[frame], cls[frame]):
            max_overl = 0.0
            track_id = -1
            for id_gt in range(num_gt):
                   
                overl = overlap(bboxes_gt[frame][id_gt], bb)
                c_gt = cls_gt[frame][id_gt]
                if overl > max_overl and c == c_gt:
                    max_overl = overl
                    track_id = track_ids[frame][id_gt]
            if max_overl > IOU_thresh:
                first_detect_pertrack[track_id] = min(first_detect_pertrack[track_id], frame)

    delays = []
    classes = []
    ids = []
    first_frames = []
    sizes = []
    for track_id in first_frame_pertrack:
        size = np.mean(size_pertrack[track_id][:30])
        first_frame = first_frame_pertrack[track_id]
        if track_id in first_detect_pertrack:
            delay = first_detect_pertrack[track_id] - first_frame_pertrack[track_id]
        else:
            delay = last_frame_pertrack[track_id] - first_frame_pertrack[track_id]
        delay = max(delay, 0)

        assert delay >= 0
        delays.append(delay)
        classes.append(class_pertrack[track_id])
        ids.append(track_id)
        first_frames.append(first_frame_pertrack[track_id])
        sizes.append(size)

    return delays, classes, ids, first_frames, sizes

def get_delays(res_files, gt_dir, conf):
    delays = []
    classes = []
    first_frames = []
    sizes = []
    tags = []
    for res_file in res_files:
        tag = res_file.split('/')[-1].split('.')[0]
        gt_file = gt_dir + '/%s.txt'%tag
        bboxes_gt, scores_gt, cls_inds_gt, track_ids, occlusions, truncations = readKITTI(gt_file, with_extra=True, classes=IMAGENETVID_CLASSES)
        bboxes, scores, cls_inds = read_results(res_file)

        bboxes, scores, cls_inds = filter_byconf(bboxes, scores, cls_inds, conf)
        delay_tmp, classes_tmp, ids, first_frames_tmp, sizes_tmp = calc_delay(bboxes_gt, scores_gt, cls_inds_gt, track_ids, bboxes, scores, cls_inds)
        delays.extend(delay_tmp)
        classes.extend(classes_tmp)
        first_frames.extend(first_frames_tmp)
        sizes.extend(sizes_tmp)
        tags.extend(map(lambda x:'{}-{}'.format(tag, x), ids))
    return np.array(delays), np.array(classes), np.array(first_frames), tags, np.array(sizes)

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
    target_precs = [0.5, 0.6, 0.7, 0.8, 0.9]
    target_fps = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    confs = []
    confs2 = []
    for target_prec in target_precs:
        idx = np.where(np.array(prec) < target_prec)[0][0]
        conf = predictions_a[idx]['confidence']
        confs.append(conf)
    for target_fp in target_fps:
        idxs = np.where(np.array(fp_rate) > target_fp)[0]
        if len(idxs) > 0:
            idx = idxs[0]
            conf = predictions_a[idx]['confidence']
        else:
            conf = 0
        confs2.append(conf)

    return mAP, confs, confs2

def get_mD(res_files, gt_dir, confs, choice='mean'):
    delays = []
    nonfirst_delays = []
    small_delays = []
    med_delays = []
    large_delays = []
    targeted_cls = [4,5,7,9,18]
    delays_per_classes = [[] for _ in targeted_cls]
    for conf in confs:
        delays_atconf, classes, first_frames, tags, sizes = get_delays(res_files, gt_dir, conf=conf)
        delay = eval_delay(delays_atconf, choice=choice)
        delays.append(delay)
        small_delays.append(eval_delay(delays_atconf[np.where(sizes < 40)[0]], choice=choice))
        med_delays.append(eval_delay(delays_atconf[np.where((sizes >= 40) * (sizes < 100))[0]], choice=choice))
        large_delays.append(eval_delay(delays_atconf[np.where(sizes >= 100)[0]], choice=choice))
        nonfirst_delay = map(delays_atconf.__getitem__, filter(lambda x: first_frames[x] != 0, range(len(delays_atconf))))
        nonfirst_delays.append(eval_delay(nonfirst_delay, choice=choice))
        for idx, cls in enumerate(targeted_cls):
            delays_per_classes[idx].append(eval_delay(delays_atconf[np.where(classes == cls)[0]], choice=choice))

    print "Number of GT instances:", len(delays_atconf)
    print "mD for small:", eval_delay(small_delays, choice='harmean')
    print "mD for med:  ", eval_delay(med_delays, choice='harmean')
    print "mD for large:", eval_delay(large_delays, choice='harmean')
    print "Nonfirst Delays:", nonfirst_delays
    print "Nonfirst frames AD:", eval_delay(nonfirst_delays, choice='harmean')

    for idx, cls in enumerate(targeted_cls):
        print "%10s AD:"%(IMAGENETVID_CLASSES[cls]), eval_delay(delays_per_classes[idx], choice='harmean')
    return delays, eval_delay(delays, choice='harmean')

def eval_delay(delays, choice):
    delays = np.array(delays).astype('f')
    if choice == 'mean':
        return np.mean(delays)
    elif choice == 'median':
        return np.median(delays)
    elif choice == 'logmean':
        return np.exp(np.mean(np.log(delays+1)))
    elif choice == 'harmean':
        return 1.0 / np.mean(1.0 / (delays+1)) - 1
    elif choice == 'clipmax':
        return np.mean(np.minimum(delays, 30))
    else:
        raise Exception

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
    print "Evaluated on %d sequences"%num_files

    split_points = np.round(np.arange(args.fold+1).astype('f') / (args.fold) * num_files).astype('i')

    for fold in range(args.fold):
        if args.fold > 1:
            print "Fold %d"%fold
        res_files = all_res_files[split_points[fold]:split_points[fold+1]]
        mAP, _, confs = get_mAP(res_files, args.gt_dir)
        print "mAP:", mAP
        choice = 'clipmax'
        delays, mD = get_mD(res_files, args.gt_dir, confs, choice=choice)
        print "Delays:", delays
        print "mD:", mD
