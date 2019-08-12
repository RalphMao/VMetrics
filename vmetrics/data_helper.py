from collections import defaultdict
import numpy as np

def read_results(filename):
    bboxes = defaultdict(list)
    scores = defaultdict(list)
    cls_inds = defaultdict(list)
    for line in open(filename):
        frame_id, cls, score, xmin, ymin, xmax, ymax = map(float, line.split())
        frame_id = int(frame_id)
        bboxes[frame_id].append((xmin, ymin, xmax, ymax))
        scores[frame_id].append(score)
        cls_inds[frame_id].append(int(cls))
    for frame_id in bboxes:
        bboxes[frame_id] = np.array(bboxes[frame_id])
        scores[frame_id] = np.array(scores[frame_id])
        cls_inds[frame_id] = np.array(cls_inds[frame_id])
    return bboxes, scores, cls_inds

def write_results(filename, bboxes, scores, cls_inds):
    with open(filename, 'w') as f:
        for idx in bboxes:
            for bbox, score, cls in zip(bboxes[idx], scores[idx], cls_inds[idx]):
                f.write('%d %d %f %d %d %d %d\n'%((idx, cls, score)+tuple(bbox)))

def writeKITTI(filename, bboxes, scores, cls_inds, track_ids=None, classes=None):
    f = open(filename, 'w')
    num_frames = len(bboxes)
    for fid in range(num_frames):
        for bid in range(len(bboxes[fid])):
            fields = [''] * 17
            fields[0] = fid
            fields[1] = -1 if track_ids is None else int(track_ids[fid][bid])
            fields[2] = classes[int(cls_inds[fid][bid])]
            fields[3:6] = [-1] * 3
            fields[6:10] = bboxes[fid][bid]
            fields[10:16] = [-1] * 6
            fields[16] = scores[fid][bid]
            fields = map(str, fields)
            f.write(' '.join(fields) + '\n')
    f.close()

def readKITTI(filename, with_extra=False, classes=None):
    bboxes = defaultdict(list)
    scores = defaultdict(list)
    cls_inds = defaultdict(list)
    track_ids = defaultdict(list)
    occlusions = defaultdict(list)
    truncations = defaultdict(list)
    for line in open(filename):
        if line.strip() == '': continue
        fields = line.split()
        assert len(fields) == 17, "File format unknown"
        frame_id = int(fields[0])
        track_id = int(fields[1])
        truncation = int(fields[3])
        occlusion = int(fields[4])
        if fields[2] == 'Person':
            cls = classes.index('Person_sitting')
        else:
            cls = classes.index(fields[2])
        score = float(fields[-1])
        xmin, ymin, xmax, ymax = map(float, fields[6:10])
        bboxes[frame_id].append((xmin, ymin, xmax, ymax))
        scores[frame_id].append(score)
        cls_inds[frame_id].append(cls)
        track_ids[frame_id].append(track_id)
        occlusions[frame_id].append(occlusion)
        truncations[frame_id].append(truncation)
    for frame_id in bboxes:
        bboxes[frame_id] = np.array(bboxes[frame_id])
        scores[frame_id] = np.array(scores[frame_id])
        cls_inds[frame_id] = np.array(cls_inds[frame_id])
        track_ids[frame_id] = np.array(track_ids[frame_id])
        occlusions[frame_id] = np.array(occlusions[frame_id])
        truncations[frame_id] = np.array(truncations[frame_id])
    if with_extra:
        return bboxes, scores, cls_inds, track_ids, occlusions, truncations
    else:
        return bboxes, scores, cls_inds
    
IMAGENETVID_CLASSES = ('__background__', 'airplane', 'antelope', 'bear', 'bicycle',
           'bird', 'bus', 'car', 'cattle',
           'dog', 'domestic_cat', 'elephant', 'fox',
           'giant_panda', 'hamster', 'horse', 'lion',
           'lizard', 'monkey', 'motorcycle', 'rabbit',
           'red_panda', 'sheep', 'snake', 'squirrel',
           'tiger', 'train', 'turtle', 'watercraft',
           'whale', 'zebra')

