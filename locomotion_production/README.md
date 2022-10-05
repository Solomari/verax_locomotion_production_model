# Locomotion Modeling

There are two parts of the locomotion modeling:

1. Use the Verax ML Pipeline to train a model to predict FH and a model to predict TDS
2. Use get_locomotion_prediction.py to combine the models to make predictions on new data

The process looks like this

``` 
python run_pipeline.py --config locomotion_production_8-5-2022.cfg

python get_locomotion_prediction.py --cl models/locomotion_production_8-5-2022/locomotion_production_8-5-2022_clfs.p
--dataset Jan2022BroilermMXData_training.csv --predict locomotion_test.csv 

```
The two pieces are explained in more detail below


# 1) Verax ML Pipeline

This is the pipeline for Verax machine learning. The current entrypoint is "run_pipline.py". This script stitches together the functions packed in "verax_pipeline" to clean and prepare data, train a GBM model, and then report statistics on the accuracy of the model. The script takes a config file as input. 

See config_template.cfg for an example.

## Install and setup

Python 3 is required to run the verax package. It was developed using python 3.10, but other python 3 versions should work. I recommend that you create a virtual environment (conda or pyenv) for the installation. Use the setup.py script to install the custom verax package and its dependencies.

``` python setup.py develop ```

### Example

Try running the pipeline with the provided example files. From the directory "example_data":

``` python ../run_pipeline.py --config example_config.cfg ```

Read the config file for more information about the models being trained. This should create a directory "example_output" with a model file, a feature importance file, a performance metrics reports file, and subdirectories with all the processed training/testing data used.

For an example of how to work with and visualize the output files, see "example_notebook.ipynb". You can launch this jupyter notebook in your browser:

``` jupyter notebook ```

Or you can render it on the command line as an hmtl, ascii, markdown etc:

``` jupyter nbconvert --to html example_notebook.ipynb ```


### Config parameter documentation:

**\[IO\]**

*training_file* : Required. Path to a csv file with verax training data. Can be in an S3 bucket or local

*testing_file* : Optional. Set to "no" if you want to automatically generate a test set from the train set. If you want a specific test set, provide a path to a csv file

*save_bucket* : Required. S3 bucket to save the results in. Should start with s3://

*save_prefix* : Required. Identifying prefix for the saved results. Can include a directory. Reccomendation: include the date in the prefix


**\[features\]**

*poc_markers* : Required. Comma delimited list of markers from "point of care" devices (ie istat, vetscan, or icheck) to use for prediction. Names must match the columns in training_file exactly. Set to "all" if you want to include the full standard set of markers. set to "none" if you don't want to include any

*device* : Required. One of `istat`, `vetscan`, `icheck`, a combination of two of the devices (ex: `istat+icheck`), or `all`

*additional_features* : Optional. Any non-poc features to also include. Set to "none" if you don't want to include any. Set to "all" if you want to include every feature. If you use "all" here it will override the poc_markers keyword / list and just use everything. Note - *everything* includes any vet metrics as well


**\[labels\]**

*labels* : Required. Comma delimited list of labels to use as the prediction outcomes. Must match columns in training_file exactly

*group_labels* : Optional. Set to "no" if you don't want to group the labels. If you do want to create groups from the labels, spell them out in this format `group_name - vetscore,vetscore,vetscore - threshold; group_name - vetscore,vetscore,vetscore - threshold`
threshold is a number that represents the number of scores in the group that must be positive to count the group as positive

**\[settings\]**

*seed* : Required. Any integer. Seed for reproducibilty

*verbose*: Required. "yes" or "no" depending on how much information you want on the console while the model is training


# 2) Get_locomotion_prediction

This script takes the FH model and the TDS model, makes a prediction on the data with each one, and then combines the prediction into a final prediction (positive/negative).

Before the script makes a prediction, it finds optimal weights for each of the models and an optimal cutoff for what will be called positive. This process is time intensive, but the final prediction is quick. In the final implementation it would make sense to calculate the optimal weights/cutoffs only once and save those. Then make predictions on multiple datasets with those same weights/cutoffs. This would involve running the "find_cutoff_and_weights" function from the script once, then the "predict_with_thresh" function multiple times using the genereated cutoffs and weights. 

## Usage

usage: get_locomotion_prediction.py [-h] --classifiers CLFS --dataset FULL_DATA [--recall TARGET_RECALL] --predict
                                    NEW_DATA

Combine a FH and TDS model for a final locomotion prediction

options:
  -h, --help            show this help message and exit
  --classifiers CLFS    Dictionary containing a FH classifier and a TDS classifier. Must be a pickle file
  --dataset FULL_DATA   Data used to calibrate the combined model (choose a cutoff and a weights for each model). Must be
                        csv format
  --recall TARGET_RECALL
                        Recall to target when calibrating the combined model. Float. Default = 0.7
  --predict NEW_DATA    Data to make a prediction on. CSV file
