import argparse, warnings, pickle
import pandas as pd
import numpy as np
from sklearn import metrics
import verax_pipeline as vp

######## Set up the command line interface ###########
parser = argparse.ArgumentParser(description='Combine a FH and TDS model for a final locomotion prediction')
parser.add_argument('--classifiers', dest='clfs', 
                    help = "Dictionary containing a FH classifier and a TDS classifier. Must be a pickle file", required = True)
parser.add_argument('--dataset', dest = 'full_data',
                    help = "Data used to calibrate the combined model (choose a cutoff and a weights for each model). Must be csv format",
                    required = True)
parser.add_argument('--recall', dest = 'target_recall', required = False,
                    help = "Recall to target when calibrating the combined model. Float. Default = 0.7")
parser.add_argument('--predict', dest = 'new_data', required = True, 
                    help = "Data to make a prediction on. CSV file")

args          = parser.parse_args()
clfs          = args.clfs
full_data     = args.full_data
target_recall = args.target_recall
if target_recall is None:
    target_recall = 0.7
new_data      = args.new_data

features = [ "vAST", "vCK", "vUA", "vGLU", "vCA", "vPHOS", "vTP", "vALB", "vGLOB", "vK", "vNA", "iNa", "iK", "iCl",
             "iTCO2", "iGlu", "iAnGap", "iHct", "iHb", "Carotenoids_whole_blood", "Age" ]
labels = ["FH", "TDS"]
SEED = 1
###### Functions ######

def find_weights(loco_df):
    """ Analyze the full set of locomotion data to find appropriate weights for the TDS and FH models
    
    Arguments:
    loco_df -- pandas dataframe. Must contain columns called 'TDS' and 'FH'
    
    Returns:
    weight for FH, weight for TDS -- a tuple of 2 floats
    
    """
    fh_tp      = loco_df["FH"].sum()
    tds_tp     = loco_df["TDS"].sum()
    fh_weight  = fh_tp / (fh_tp + tds_tp)
    tds_weight = tds_tp / (fh_tp + tds_tp)
    
    return fh_weight, tds_weight

def find_cutoff_and_weights(loco_df, clf_dict, target_tpr):
    """ Make a prediction on the given data with the given classifiers and weights
    
    Arguments:
    loco df    -- pandas dataframe.  Must contain columns called 'TDS', 'FH', and 'LOCO' (the sum of TDS and FH)
    clf_dict   -- Dictionary. Must contain a FH model (clf_dict['FH']) and a TDS model (clf_dict['TDS'])
    target_tpr -- Float. True positive rate AKA recall to target with the cutoff tuning
    
    Returns:
    fh_weight, tds_weight, thresh -- a tuple of 3 floats. The weights for FH and TDS and the threshold for calling a positive
    
    """
    fh_weight, tds_weight = find_weights(loco_df)
    X = loco_df.drop(["TDS", "FH", "LOCO"], axis = 1)
    full_preds = X.apply(lambda x: predict_loco(clf_dict, x, fh_weight, tds_weight), axis = 1, result_type = 'expand')
    full_true  = [ 1 if lab >= 1 else 0 for lab in loco_df["LOCO"] ]
    fpr, tpr, thresholds = metrics.roc_curve(full_true, full_preds, pos_label=1)
    metric_df = pd.DataFrame({ "fpr" : fpr, "tpr" : tpr, "threshold" : thresholds})
    # get the highest threshold that gives specified recall aka TPR
    thresh = metric_df.loc[metric_df["tpr"] >= target_tpr, "threshold"].iloc[0]
    return fh_weight, tds_weight, thresh

def predict_loco(clf_dict, X, FH_weight, TDS_weight):
    """ Make a prediction on the given data with the given classifiers and weights
    
    Arguments:
    clf_dict   -- Dictionary. Must contain a FH model (clf_dict['FH']) and a TDS model (clf_dict['TDS'])
    X          -- numpy array. one sample to make a prediction on
    FH_weight  -- float. Weight to give the FH model prediction
    TDS_weight -- float. Weight to give the TDS model prediction
    
    Returns:
    Float with a prediction for the data
    
    """
    X = np.reshape(X.to_numpy(), (1, -1))
    FH_clf, TDS_clf  = clf_dict["FH"], clf_dict["TDS"]
    
    FH_pred, TDS_pred = FH_clf.predict_proba(X)[0,1], TDS_clf.predict_proba(X)[0,1]
    return (FH_pred * FH_weight) + (TDS_pred * TDS_weight)

def predict_with_thresh(clf_dict, X, fh_weight, tds_weight, thresh):
    """ Make a prediction on the given data with the given classifiers and weights
    
    Arguments:
    clf_dict   -- Dictionary. Must contain a FH model (clf_dict['FH']) and a TDS model (clf_dict['TDS'])
    X          -- numpy array. one sample to make a prediction on
    FH_weight  -- float. Weight to give the FH model prediction
    TDS_weight -- float. Weight to give the TDS model prediction
    thresh     -- float. Threshold for what to call a positive
    
    Returns:
    Int representing binary prediction for the data
    
    """
    preds = X.apply(lambda x: predict_loco(clf_dict, x, fh_weight, tds_weight), axis = 1, result_type = 'expand')
    bin_preds = [ 1 if pred >= thresh else 0 for pred in preds ]
    return bin_preds
    
### Run the predictions    
clf_dict = pickle.load(open(clfs, 'rb'))

loco_df = vp.check_ext(full_data, features, labels)
loco_df["LOCO"] = loco_df["TDS"] + loco_df["FH"]
new_df = vp.check_ext(new_data, features, labels)
new_d = vp.preprocess_data("TDS", new_df, verbose = True, elim_cols = False, elim_missing_labels = False) 
X_train, X_test, _, _, _ = vp.split_train_test_impute(new_d, "TDS", SEED, features, False)
X = pd.concat([X_train, X_test])

fh_weight, tds_weight, thresh = find_cutoff_and_weights(loco_df, clf_dict, target_recall)
print(predict_with_thresh(clf_dict, X, fh_weight, tds_weight, thresh))