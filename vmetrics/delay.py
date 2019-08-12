from collections import defaultdict
import numpy as np

from .utils import overlap

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


