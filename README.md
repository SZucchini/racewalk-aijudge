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
### Step 0: Download the required data

Please download data that you need from [Google Drive](https://drive.google.com/drive/folders/1BbYuti87mX995lcWFvLyYF_edIehQjNB?usp=sharing).
- **`interim`**: You can download keypoint data (`keypoints`) and faults annotation result (`judge_result.csv`).
- `raw/video`: There are raw videos we captured. If you want to get keypoints from videos by yourself, please use them.
- `external/mmpose_materials`: You can download fine-tuned model weights (`model`) and images for fine-tuning HigherHRNet (`dataset/annotations/img`) if you need.

## Judgment model evaluation
### Step1: Make the input data for the judgment model (You can skip this step).
- `cd scripts`
- `python make_input.py`
- Then the model input data is output to `./data/processed/hrf/*.csv`.

### Step2: Evaluation of the judgment model and visualization of the judgment grounds.
- Please open the notebook `./notebooks/*.ipynb`
- Run all cells and check the result.
