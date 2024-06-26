# [Automatic detection of faults in race walking from a smartphone camera](https://arxiv.org/abs/2208.12646)

Tomohiro Suzuki, Kazuya Takeda, Keisuke Fujii, [Automatic Fault Detection in Race Walking From a Smartphone Camera via Fine-Tuning Pose Estimation](https://ieeexplore.ieee.org/document/10014142/), IEEE 11th Global Conference on Consumer Electronics (GCCE 2022), 2022.10.20. (Gold Prize in IEEE GCCE 2022 Excellent Student Paper Awards) [\[Full(arXiv)\]](https://arxiv.org/abs/2208.12646) [\[code\]](https://github.com/SZucchini/racewalk-aijudge/tree/main/notebooks)

## Introduction

This is the official code of [Automatic detection of faults in race walking from a smartphone
camera](https://arxiv.org/abs/2208.12646).

We can detect faults in race walking automatically from video using the code and [mmpose](https://github.com/open-mmlab/mmpose). The accuracy of our detection models is over 90%. You can try to train and validate models using raw video data or processed data.

This repository includes:
- a script for creating training and validation data for the faults judgment model.
- two notebooks for evaluating the faults judgment models.

This repository **does not include**:
- [mmpose](https://github.com/open-mmlab/mmpose) project. (We used the [higherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) through mmpose to estimate keypoints from videos.)

If you have any questions or errors, please contact to the author.

## Sample Image

![sample](https://user-images.githubusercontent.com/78769319/201851811-6267d759-171c-42bd-b3f4-67bf0272c757.jpg)

## Author
Tomohiro Suzuki ([Takeda, Fujii Laboratory](https://takedalab.g.sp.m.is.nagoya-u.ac.jp/groups/sports-behavior-group), Nagoya University) - suzuki.tomohiro[at]g.sp.m.is.nagoya-u.ac.jp

## Environment
- We used the Anaconda environment.
- Please check `env.yaml` and `env.frozen.yaml`.
- `$ conda env create --file env.frozen.yaml`

## Evaluation from scratch
### Step 0: Downloading the required data

Please download data that you need from [Google Drive](https://drive.google.com/drive/folders/1BbYuti87mX995lcWFvLyYF_edIehQjNB?usp=sharing).
- **`processed`**: You can download input data for fault detection models. If you use these data, you can skip to Step 3.
- **`interim`**: You can download keypoint data (`keypoints`) and faults annotation result (`judge_result.csv`). If you use these data, you can skip to Step 2.
- `raw/video`: There are raw videos we captured. If you want to get keypoints from videos by yourself, please use them. We obtained permission to release raw videos from walker B, C, D, E. However, walker B has a blurred face because he wished to anonymize his data. About walker A, we could not obtain his permission.
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

## Sample Video (can be played with Safari)

![demo](https://user-images.githubusercontent.com/78769319/201831462-99f21272-3fdf-4105-b868-982619f30d1f.mp4)

## Citation
If you find our work useful for your project, please consider citing these papers:
```
@article{suzuki2022automatic,
  title={Automatic detection of faults in race walking from a smartphone camera: a comparison of an Olympic medalist and university athletes},
  author={Suzuki, Tomohiro and Takeda, Kazuya and Fujii, Keisuke},
  journal={arXiv preprint arXiv:2208.12646},
  year={2022}
}

@inproceedings{suzuki2022automatic,
  title={Automatic fault detection in race walking from a smartphone camera via fine-tuning pose estimation},
  author={Suzuki, Tomohiro and Takeda, Kazuya and Fujii, Keisuke},
  booktitle={2022 IEEE 11th Global Conference on Consumer Electronics (GCCE)},
  pages={631--632},
  year={2022},
  organization={IEEE}
}
```
