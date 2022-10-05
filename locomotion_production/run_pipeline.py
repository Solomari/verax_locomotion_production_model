import pandas as pd
import sklearn, random, sys, argparse
import configparser
from botocore.exceptions import ClientError
import boto3
import numpy as np
import verax_pipeline as vp
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Train models')
parser.add_argument('--config_file', dest='config_file', 
                    help = 'File with configurations for experiment', required = True)
 
#### Parse the config file ####
args         = parser.parse_args()
config_file  = args.config_file
config       = configparser.ConfigParser()
config.read(config_file)

training_file = config["IO"]["training_file"]
testing_file  = config["IO"]["testing_file"]
save_bucket   = config["IO"]["save_bucket"]
save_prefix   = config["IO"]["save_prefix"]
poc_markers   = [ m.strip() for m in config["features"]["poc_markers"].split(",") ]
add_features  = [ af.strip() for af in config["features"]["additional_features"].split(",") ]
devices       = config["features"]["device"].strip()
labels        = [ l.strip() for l in config["labels"]["labels"].split(",") ]
group_label_dict = vp.parse_group_labels(config["labels"]["group_labels"])
SEED             = int(config["settings"]["seed"])
verbose_filter   = config["settings"]["verbose"].lower() == "yes"

# make sure the s3 bucket you're trying to write to exists before proceeding
if save_bucket.upper() != "LOCAL":
    s3 = boto3.resource('s3')
    try:
        s3.meta.client.head_bucket(Bucket=save_bucket)
    except ClientError:
        # The bucket does not exist or you have no access.
        print(f"\nError: Either 's3://{bucket}' does not exist or you do not have access\n")
        quit()
        
# get the feature list based on device given and / or feature keywords
features = vp.select_device_markers(poc_markers, add_features, devices)
    
print('================================')

big_divider = "======================================================================================="

###### read the training and testing data 

# read testing data
if testing_file.lower() == "no":
    test_data = False
else:
    test_data = vp.check_ext(testing_file, features, labels)
    if group_label_dict:
        try:
            test_data, _ = vp.group_observations(test_data, group_label_dict)
        except KeyError as e:
            print(f"Double check that all labels in group_labels are also in labels for TEST data. Error: {e}")
            quit()
# read training data
data = vp.check_ext(training_file, features, labels)
# this is the case where "all" columns were requested
if features == None:
    features = list(data.columns)

# group the labels
if group_label_dict:
    try:
        data, g_labels = vp.group_observations(data, group_label_dict)
        labels = g_labels + labels
    except KeyError as e:
        print(f"Double check that all labels in group_labels are also in labels for TRAIN data. Error: {e}")
        quit()

def run_pipeline(label, data, SEED, features, verbose = False, test_data = False):
    d = vp.preprocess_data(label, data, verbose = False)
    if isinstance(test_data, pd.DataFrame):
        d = vp.preprocess_data(label, data, verbose = True, elim_cols = False)
        test_d = vp.preprocess_data(label, test_data, verbose, elim_cols = False)
    else:
        test_d = test_data
    if len(d) < 10:
        print(f"Not enough data in the train set, skipping {label}")
        print(big_divider)
        raise ValueError
        
    try:
        X_train, X_test, y_train, y_test, estimate = vp.split_train_test_impute(d, label, SEED, features, test_d)
        print(vp.write_train_test_data(X_train, y_train, save_bucket, save_prefix, label, data_type = "train"))
        print(vp.write_train_test_data(X_test, y_test, save_bucket, save_prefix, label, data_type = "test"))
    except ZeroDivisionError:
        print(f"Not enough data to create a test set, skipping {label}")
        print(big_divider)
        raise ValueError
    except:
        print("Unexpected error:", sys.exc_info()[0], sys.exc_info()[1])
        print(f"d: {d}")
        print(f"test_data: {test_d}")
        raise
    print(f"Training data size: {len(X_train)}, Test data size: {len(X_test)}")
        
    # I've decided there should be at least 10 examples of each class in the test set to get worthwhile results
    # if len(Y_test.value_counts()) < 2, that means there's only one class represented in the test set at all
    if len(y_test.value_counts()) < 2 or y_test.value_counts().min() <10:
        print(f"Not enough data to create a test set, skipping {label}")
        print(big_divider)
        raise ValueError
    
    print("--------------------")
    try:
        clf, report = vp.train_and_score(X_train, X_test, y_train, y_test, estimate, label, SEED)
        imp_df      = vp.report_importances(X_train, clf, combine_na = True)
        print(big_divider)
        return imp_df, report, clf
    except:
        print("Unexpected error in train and score:", sys.exc_info()[0], sys.exc_info()[1])
        raise
    print(big_divider)

clf_dict    = {}
report_dict = {}
imp_list    = []

for label in labels:
    try:
        importances, report, clf = run_pipeline(label, data, SEED, features, verbose_filter, test_data)
    except ValueError:
        continue
    importances["outcome"]   = label
    clf_dict[label]          = clf 
    report_dict[label]       = report
    imp_list.append(importances)

vp.write_pickle(save_bucket, save_prefix + "/" + save_prefix.split("/")[-1] + "_clfs.p", clf_dict)
vp.write_pickle(save_bucket, save_prefix + "/" + save_prefix.split("/")[-1] + "_reports.p", report_dict)
vp.write_pickle(save_bucket, save_prefix + "/" + save_prefix.split("/")[-1] + "_importances.p", imp_list)
print(f"Saved the classifier, reports, and importances to {save_bucket}/{save_prefix}")  
