# SimCom: An Effective Just-In-Time Defect Predictor by Combining Simple and Complex Models

## Dependency

Python >=3.6.9

pip install torch

pip install transformers

## data & pre-trained models

data can be found here (https://drive.google.com/file/d/1WbWC2lhHLW16OCycV4yLzIF9S4dLb6om/view?usp=sharing)

In the "data" folder, "commit_content" is for the Complex Model and "hand_crafted_features" is for the Simple Model.

## Run SimCom

    $$ python run_simcom.py -project [project_name]

Example:

    $$ python run_simcom.py -project qt

Note that: the training (validation) and testing sesstions are integrated into a single script "run_simcom.py". It will first train and do predictions on the test sets.


