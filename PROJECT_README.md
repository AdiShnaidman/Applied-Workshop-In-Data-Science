# Project: Competitive-Lab
Authors: Ben Moyal, Adi Shnaidman, Daniel Ohana & Avraham Meisel

This project contains several files within the src code directory:

1. preprocess.py
This file houses all preprocessing functions. It includes a main function that preprocesses the data, tailored for either model training or graph presentation. Each pipeline imports its main function to preprocess the data.

2. visualization.ipynb
This notebook preprocesses the data, customized for the features we aim to present. It uses the main function from preprocess.py and displays all the graphs included in the report.

3. models_selection.ipynb
In this notebook, we train various models with different hyperparameters. The best model is selected based on the AUC-ROC metric. Please refer to the notes section for information about the Neural Network (NN).

4. chosen_model_pipeline.ipynb
This notebook takes the best model from models_selection_pipline.ipynb and performs feature selection.
# This is the model we recommend for testing.

# CSV Files
Ensure that the following CSV files are located in the src code directory:

data.csv
data_test.csv
validation.csv
To view the results, install the required libraries in your environment by running:
```
pip install -r requirements.txt
```

Then, run any notebook of your choice.

# Notes
We attempted to run a Neural Network (NN) model, but due to lack of focus on such models this semester and the resulting performance issues, we have commented it out.

# Instructions for Creating a Virtual Environment
Open your terminal or command prompt. You can do this by searching for “Terminal” on macOS or Linux, or “Command Prompt” on Windows.
Create a virtual environment named env by running:
```
python -m venv env_name
```

Activate the Virtual Environment:
On Windows:
```
path_to_env\env\Scripts\activate
```
On macOS or Linux:
```
source path_to_env/env/bin/activate
```
After activating it, run the command:
```
pip install -r requirements.txt
```
Ensure your kernel is using this environment while running the code. Enjoy :)