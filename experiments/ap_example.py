import glob
import sys

sys.path.insert(0, './')

from vmetrics.utils import read_gt_files, read_pred_files
from vmetrics.ap import eval_mAP

# get a list with the ground-truth files
ground_truth_files_list = glob.glob('examples/ground-truth/*.txt')
ground_truth_files_list.sort()
groundtruths = read_gt_files(ground_truth_files_list)
# get a list with the predicted files
predicted_files_list = glob.glob('examples/predicted/*.txt')
predicted_files_list.sort()
predictions = read_pred_files(predicted_files_list)

mAP, _ = eval_mAP(groundtruths, predictions)
print "mAP: %.4f"%mAP

