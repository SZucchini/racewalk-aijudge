# Coming soon...

## Introduction

This is the official code of [Automatic detection of faults in race walking from a smartphone
camera]().

We can detect faults in race walking automatically using the code and [mmpose](https://github.com/open-mmlab/mmpose). The accuracy of our detection models is over 90%. You can try to train and validate models using raw video data or processed data.

This repository includes:
- a script for creating training and validation data for the faults judgment model.
- two notebooks for evaluating the faults judgment models.

This repository **does not include**:
- [mmpose](https://github.com/open-mmlab/mmpose) project. (We used the [higherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) through mmpose to estimate keypoints from videos.)

If you have any questions or errors, please contact to the author.

## Author
Tomohiro Suzuki - suzuki.tomohiro@g.sp.m.is.nagoya-u.ac.jp

## Environment
- We used the Anaconda environment.
- Please check `env.yaml` and `env.frozen.yaml`.

## Evaluation from scratch
### Step 0: Downloading the required data

Please download data that you need from [Google Drive](https://drive.google.com/drive/folders/1BbYuti87mX995lcWFvLyYF_edIehQjNB?usp=sharing).
- **`processed`**: You can download input data for fault detection models. If you use these data, you can skip to Step 3.
- **`interim`**: You can download keypoint data (`keypoints`) and faults annotation result (`judge_result.csv`). If you use these data, you can skip to Step 2.
- `raw/video`: There are raw videos we captured. If you want to get keypoints from videos by yourself, please use them.
- `external/mmpose_materials`: You can download fine-tuned model weights (`model`) and images for fine-tuning HigherHRNet (`dataset/annotations/img`) if you need.

### Step 1: Estimating keypoints from walking videos

Please check and use [mmpose](https://github.com/open-mmlab/mmpose) to estimate keypoints. You can use our custom scripts  and data for mmpose in [mmpose_materials](https://github.com/SZucchini/racewalk-aijudge/tree/main/data/external/mmpose_materials).

### Step 2: Making model input data

1. Please put `judge_result.csv` into `./data/interim` and keypoints data into `./data/interim/keypoints/`.
1. `cd scripts`
1. `python make_input.py`
1. Then, input data is output to `./data/processed`.

### Step 3: Evaluation of the models and visualization of the detection grounds

1. Please put `*_input_data.csv` into `./data/processed` if you start from this step.
1. Please open the notebook `./notebooks/*.ipynb`.
1. Run all cells and check the results.
