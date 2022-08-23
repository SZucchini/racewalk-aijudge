# Race Walk Judge

## Abstract
This project allows for the verification of non-contact fault judgment in race walking.
See [Documents](https://drive.google.com/drive/folders/1maZTWzhs-6zKQgCuP18GBr1YHXQJZKtT?usp=sharing) for more information on non-contact fault judgment.

This project includes:
- the script for creating training and validation data for the judgment model (Logistic regression).
- two notebooks for evaluating the fault judgment models.
- some data (keypoints estimated by pose estimator and model input).

This project **does not include**:
- [mmpose](https://github.com/open-mmlab/mmpose) project (We used the [higherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) by mmpose to estimate keypoints from videos.).
- You can download our custom materials for the mmpose from [here](https://drive.google.com/drive/folders/1zTtPlkrcJdDwPtaGnINEeIa5FW43nMaV?usp=sharing).

If you have any questions or errors, please contact the author.

## Author
Tomohiro Suzuki - suzuki.tomohiro@g.sp.m.is.nagoya-u.ac.jp

## Requirements
- We used the Anaconda environment.
- Please check `env.yaml` and `env.frozen.yaml`.

## Judgment model evaluation
### Step1: Make the input data for the judgment model (You can skip this step).
- `cd scripts`
- `python make_input.py`
- Then the model input data is output to `./data/processed/hrf/*.csv`.

### Step2: Evaluation of the judgment model and visualization of the judgment grounds.
- Please open the notebook `./notebooks/*.ipynb`
- Run all cells and check the result.
