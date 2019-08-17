# VMetrics

This repo provides the evaluation codes used in our ICCV 2019 paper [A Delay Metric for Video Object Detection: What Average Precision Fails to Tell](), including:

- Mean Average Precision (mAP)
- Average Delay (AD)
- A redesigned [NAB](https://github.com/numenta/NAB) metric for the video object detection problem.

### Prepare the data

Download the groundtruth annotations and the sample detector outputs by running the following command:

```sh
$ bash prep_data.sh
```

The groundtruth annotations of VIDT are stored in KITTI-format due to its simplicity and io-efficiency.

We provide the outputs of the following methods. The github repos that generate those outputs are also listed.

### Run evaluation

All the evaluation scripts are under `./experiments` folder. For instance, to measure the mAP and AD of FGFA, run command:

```
python experiments/eval_map_ad.py examples/rfcn_fgfa_7 data/ILSVRC2015_KITTI_FORMAT
```

### Evaluate your own detector.

For every video sequence, output a file as `<sequence_name>.txt`. Each line in the file should be one single object in `<frame_id> <class_id> <confidence> <xmin> <ymin> <xmax> <ymax>` format.

### Acknowledgement

This pure Python-based mAP evaluation code is refactored from [Cartucho/mAP](https://github.com/Cartucho/mAP). It has been tested against the original matlab version.
