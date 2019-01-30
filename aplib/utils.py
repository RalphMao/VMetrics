
import os
import sys
from collections import defaultdict

def error(msg):
    print(msg)
    sys.exit(0)

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

def read_gt_files(ground_truth_files_list, ignores=None):
    if ignores is None:
        ignores = []
    groundtruths = defaultdict(lambda: defaultdict(list))
    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
                    is_difficult = False
                bbox = tuple(map(float, (left, top, right, bottom)))
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)

            # check if class is in the ignore list, if yes skip
            if class_name in ignores:
                continue
            groundtruths[class_name][file_id].append({"bbox":bbox, "difficult":is_difficult})
    return groundtruths

def read_pred_files(predicted_files_list):
    predictions = defaultdict(list)
    for txt_file in predicted_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))

        lines = file_lines_to_list(txt_file)
        for line in lines:
            try:
                class_name, confidence, left, top, right, bottom = line.split()
                bbox = tuple(map(float, (left, top, right, bottom)))
                confidence = float(confidence)
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                error_msg += " Received: " + line
                error(error_msg)
            predictions[class_name].append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

    for class_name in predictions:
        predictions[class_name].sort(key=lambda x:x['confidence'], reverse=True)
    return predictions

def rcnn2gt(bboxes, scores, cls_inds):
    groundtruths = defaultdict(lambda: defaultdict(list))
    for file_id in bboxes:
        for idx in range(len(bboxes[file_id])):
            class_name = int(cls_inds[file_id][idx])
            bbox = bboxes[file_id][idx]
            groundtruths[class_name][file_id].append({"bbox":bbox, "difficult":False})
    return groundtruths

def rcnn2pred(bboxes, scores, cls_inds):
    predictions = defaultdict(list)
    for file_id in bboxes:
        for idx in range(len(bboxes[file_id])):
            class_name = int(cls_inds[file_id][idx])
            bbox = bboxes[file_id][idx]
            confidence = scores[file_id][idx]
            predictions[class_name].append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

    for class_name in predictions:
        predictions[class_name].sort(key=lambda x:x['confidence'], reverse=True)
    return predictions

def rcnn2apred(bboxes, scores, cls_inds):
    '''
    This func is intended for eval_aAP
    '''
    predictions = []
    for file_id in bboxes:
        for idx in range(len(bboxes[file_id])):
            class_name = int(cls_inds[file_id][idx])
            bbox = bboxes[file_id][idx]
            confidence = scores[file_id][idx]
            predictions.append({"confidence":confidence, "file_id":file_id, "bbox":bbox, "class":class_name})

    predictions.sort(key=lambda x:x['confidence'], reverse=True)
    return predictions

