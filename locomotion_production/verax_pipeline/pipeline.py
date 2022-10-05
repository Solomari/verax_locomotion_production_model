import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from collections import Counter
from sklearn.metrics import classification_report
import warnings

def preprocess_data(label, data, verbose, elim_cols = True, elim_missing_labels = True):
    """ Process the data for modeling. This is the general processing that is done before the train/test split
    
    Arguments:
    label      -- String. The label that will be the outcome for the classifer.
    data       -- pandas dataframe. data to preprocess
    verbose    -- boolean. do you want to know how many rows/columns are dropped?
    elim_cols  -- boolean. Drop features with lots of missing data (>70%)? Recommended to set to True
    
    Returns:
    dataframe where month and sex are coded as ordinals, labels are coded as 0 or 1, samples with lots of missing data are dropped
    
    """
    print("Training: " + label)
    
    # if month or sex are in the columns, code it numerically
    month_mapper = { "January" : 1, "February" : 2, "March" : 3, "April" : 4, "May" : 5,
                    "June" : 6, "July" : 7, "August" : 8, "September" : 9, "October" : 10,
                    "November" : 11, "December" : 12 }
    sex_mapper = { "M" : 0, "F" : 1 }

    if "Month" in data and data.dtypes["Month"] == np.object:
        data["Month"] = data["Month"].replace(month_mapper)
    if "Sex" in data and data.dtypes["Sex"] == np.object:
        data["Sex"]   = data["Sex"].replace(sex_mapper)
    # if any non-numeric columns are left, get rid of them
    num_data = data.copy().select_dtypes(['number'])
    if len(num_data.columns) != len(data.columns):
        warnings.warn(f"Dropping {len(data.columns) - len(num_data.columns)} non-numeric features that you asked to include.",
                      UserWarning)
        data = num_data
    
    # Make all positive values with different severities = 1
    data.loc[data[label] >= 1, label] = 1.0
    
    # Drop the rows where this label is an empty entry
    if elim_missing_labels:
        d = data[data[label].notna()]
    else:
        d = data
    if verbose:
        print(f"Eliminated {len(data) - len(d)} samples where {label} score is missing. " +
              f"{len(d)} ({len(d) / len(data) * 100:.1f}%) samples remain.")
    
    if elim_cols:
        # Drop the columns where data is (70%) missing
        col_perc_thresh = 0.7
        col_frac = len(d) * col_perc_thresh
        d = d.dropna(thresh=col_frac, axis=1)
        if verbose:
            elim_columns = [ col for col in list(data.columns) if col not in list(d.columns) ]
            print(f"Eliminated {len(elim_columns)} features that were > {col_perc_thresh * 100}% " +
                  f"missing values : {elim_columns}. {len(d.columns)} " + 
                  f"({len(d.columns) / len(data.columns) * 100:.1f}%) features remain.")
    
    # Drop the rows where data is (80%) missing in blood biomarkers
    perc_thresh = 0.8
    frac = len(d.columns) * perc_thresh
    len_first_cut = len(d) 
    d= d.dropna(thresh=frac, axis=0)
    if verbose:
        print(f"Eliminated {len_first_cut - len(d)} samples that were missing > {perc_thresh * 100}%  of their " +
              f"biomarker measurements. {len(d)} ({len(d) / len(data) * 100:.1f}%) samples remain.")
   
    return d

def impute_scale(X, scale = False):
    """ Impute missing values and scale the features (optional)
    
    Arguments:
    X      -- pandas dataframe with predictive features
    scale       -- boolean. Use standard scalar or not? Current preference is to leave data unscaled
    
    Returns:
    dataframe where missing values are imputed using KNN imputer. optionally the data will also be scaled with standard scaler
    """
    all_na = X.loc[:, X.isna().all()].columns
    if len(all_na) > 0:
        print(f"Some biomarkers have no measurements, cannot impute for these: {all_na}. Try setting elim_cols = True")
        raise
        
    imp = KNNImputer()
    X_imp = pd.DataFrame(imp.fit_transform(X), columns = X.columns)

    if scale:
        scaler = StandardScaler()
        X_proc = pd.DataFrame(scaler.fit_transform(X_imp), columns= X.columns)
    else:
        X_proc = X_imp

    return X_proc

def count_labels(Y_train):
    """ Check the number of positives / negatives in the list of labels
    
    Arguments:
    Y_train    -- pandas series of labels, 0 or 1 
    
    Returns:
    the number of negatives / number of positives. Also prints stats to the console
    
    """
    # Count Negative/Positive entries
    counter = Counter(Y_train)
    print(f'Negative value #: {counter[0]}')
    print(f'Positive value #: {counter[1]}')
    try:
        estimate = counter[0] / counter[1]
    except ZeroDivisionError:
        print("No positives")
        return np.Inf
    print('Scale Pos Weight Estimate: %.3f' % estimate)
    print()
    return estimate

def split_train_test_impute(preprocessed_data, label, seed, markers, test_data = False):
    """ Wrapper function to impute, split the data, and count the labels
    
    Arguments:
    preprocessed_data -- dataframe that has been processed using preprocess_data
    label             -- string, outcome label
    seed              -- int
    markers           -- features to use to predict
    test_data         -- if you want to test using specific dataset, provide it here. otherwise data will be split
 
    Returns:
    X_train, X_test, Y_train, Y_test (pandas dfs / series), estimate (float : negative / positive outcomes)
    
    """
    cols_to_keep = markers + [label]
    # Use & to select the columns that are in the list AND exist (some have been dropped)
    preprocessed_data = preprocessed_data[list(set(preprocessed_data.columns).intersection(set(cols_to_keep)))]

    if isinstance(test_data, pd.DataFrame):
        X_train = preprocessed_data.drop(label, axis = 1)
        Y_train = preprocessed_data[label]
        X_test = test_data[list(set(test_data.columns).intersection(set(markers)))]
        Y_test = test_data[label]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(preprocessed_data.drop(label, axis = 1), preprocessed_data[label],
                                                            test_size=0.2, random_state=seed)
    
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    X_train  = impute_scale(X_train)
    X_test   = impute_scale(X_test)
    estimate = count_labels(Y_train)
    
    return X_train, X_test, Y_train, Y_test, estimate

def train_and_score(X_train, X_test, Y_train, Y_test, estimate, label, SEED):
    """ Train the model, get performance metrics
    
    Arguments:
    X_train  -- dataframe of processed training data
    X_test   -- dataframe of processed testing data
    Y_train  -- series of processed training labels
    Y_test   -- series of processed testing labels
    estimate -- proportion of negative labels / positive labels
    label    -- name of the outcome measured in Y
    seed     -- int
 
    Returns:
    sklearn style trained XGBoost classifier, dictionary with performance metrics
    
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
    parameter_grid = {'scale_pos_weight': estimate, 'eval_metric': 'error',
                      'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.01, 
                      'use_label_encoder' : False }
    warnings.filterwarnings(action='ignore', category=UserWarning)  
    clf = XGBClassifier(**parameter_grid, random_state = SEED)
    clf.fit(X_train, Y_train, eval_set = [(X_train, Y_train), (X_test, Y_test)], verbose = False)
    
    Y_pred = clf.predict(X_test)
    print(f"Test set size (predictions): {len(Y_pred)}")
    print(f"Test set size (labels): {len(Y_test)}")
   
    # read the train and test accuracy from the classifier
    evals_result = clf.evals_result()   
    print(f"TP rate: {Y_test.sum() / len(Y_test)}")
    display_train = f'Train accuracy, for {label}:'
    display_test  = f'Test accuracy, for {label}:'
    print(f"{display_train}: {(1 - evals_result['validation_0']['error'][-1])*100:.1f}%")
    print(f"{display_test}: {(1 - evals_result['validation_1']['error'][-1])*100:.1f}%")
    print(accuracy_score(Y_pred, Y_test))

    report = classification_report(Y_test , Y_pred, output_dict = True)
    report["train_data"] = { "0.0" : Y_train.value_counts()[0], "1.0" : Y_train.value_counts()[1] }
    
    
    Y_pred_prob = clf.predict_proba(X_test)
    return clf, report

def report_importances(X_train, clf, combine_na = True):    
    """ Make a table of the feature importances. Return the importances formatted for plotting
    
    Arguments:
    X_train    -- the pandas dataframe used to train the classifier
    clf        -- an sklearn type classifier
    combine_na -- boolean, report (for ex.) Carotenoids and Carotenoids_na together or separately?
    
    Returns:
    importances df
    
    """

    imp_df = pd.DataFrame({"feature"    : X_train.columns,
                           "importance" : clf.feature_importances_})
    imp_df = imp_df.loc[imp_df["importance"] > 0, :]
    # we can combine the 'NA' (no value reported) features with their 
    # corresponding numeric features - this gives less information but 
    # is easier to interpret
    if combine_na:
        imp_df["feature"] = imp_df["feature"].str.split("_").apply(lambda x: x[0])
        imp_df = imp_df.groupby("feature").mean().reset_index()
        
    imp_df = imp_df.sort_values(by = "importance")
    # return the df for plotting
    return imp_df
