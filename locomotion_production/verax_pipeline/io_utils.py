import pickle, io, boto3
import numpy as np
import pandas as pd
import awswrangler as wr
from pathlib import Path

def check_ext(file, features, labels):
    """ check extension of a file and read it accordingly
    
    Arguments:
    file     -- a csv or tsv file
    features -- a list of feature names that correspond to columns in the file
    labels   -- a list of labels / outcomes that correspond to columns in the file
    
    Returns:
    a pandas dataframe  
    """
    if ".csv" in file:
        sep = ","
    elif "txt" or "tsv" in file:
        sep = "\t"
    else:
        sys.exit("Unrecognized filetype. Accepts .csv or .txt (tab-delimited) files.")
    if file[0:2] == "s3":
        data = wr.s3.read_csv(file, sep = sep, low_memory = False)
    else:
        data = pd.read_csv(file, sep = sep, low_memory = False)
    # replace certain illegal characters with underscores...the characters are : [], <>=
    data.columns = data.columns.str.replace(r"[\[\]\,\ \<\>\=]", "_", regex = True)

    if features != None:
        features = [ f for f in features + labels if f in data.columns ]
        data = data[features]
        
    return data

def parse_group_labels(group_labels):
    """ Parse the group label instructions from the config file
    
    Arguments:
    group_labels -- a string read from the config file. See README or config_template.cfg for specifications
    
    Returns:
    a nested dictionary that will be used by group_observations to create new labels 
    """
    if group_labels.lower() == "no":
        group_label_dict = False
    else:
        group_label_dict = {}
        for group in group_labels.split(";"):
            name   = group.split("-")[0].strip()
            labs   = group.split("-")[1].strip()
            thresh = int(group.split("-")[2].strip())

            name_dict = { "labels" : [ l.strip() for l in labs.split(",") ], "threshold" : thresh }
            group_label_dict[name] = name_dict
    return group_label_dict

def group_observations(data, group_obs_dict):
    """ Group related vet observations to create new labels
    
    Arguments:
    data           -- a pandas dataframe that includes observations (biomarkers) and labels
    group_obs_dict -- a dictionary of dictionaries. The 'outer' keys are the names of the groups. 
                      The 'inner' dictionaries have 2 keys, 'labels' and 'threshold.' Labels are
                      the labels to include in the group, threshold is the number of labels in the
                      group that must be positive to call the group positive
    
    Returns:
    grouped_data (pandas dataframe), group_names (list)  
    
    """
    df = data.copy()
    for grouping in group_obs_dict:
        grouping_dict = group_obs_dict[grouping]
        group_labels  = grouping_dict["labels"]
        threshold     = grouping_dict["threshold"]
    
        # Make all positive values with different severities = 1
        for label in group_labels:
            df.loc[df[label] >= 1, label] = 1.0
        df[grouping]   = df[group_labels].sum(axis = 1).apply(lambda x: 1 if x >= threshold else 0)
      
    return df, list(group_obs_dict.keys())

def select_device_markers(poc_markers, add_features, devices):
    """  select the markers based on devices parameter
    
    Arguments:
    poc_markers  -- a list of point-of-care biomarkers to select from. Could also be "all" or "none"
    add_features -- a list of additional features to include. Could also be "all" or "none"
    devices      -- a string with devices to use biomarkers from. Devices could be 'vetscan', 'istat', 'icheck', 
                    'all', or a combination

    Returns:
    a list of features
    """
    if type(poc_markers) is str:
        poc_markers = [poc_markers]
    all_poc_markers = ["vAST","vBA","vCK","vUA","vGLU","vCA","vPHOS","vTP","vALB","vGLOB","vK","vNA","iNa","iK","iCl",
                       "iTCO2", "iBUN","iCrea","iGlu","iiCa","iAnGap","iHct","iHb","Carotenoids_whole_blood"]
    if poc_markers[0].lower() == "all":
        poc_markers = all_poc_markers
    elif poc_markers[0].lower() == "none":
        poc_markers  = ["none"]
    
    # filter based on devices
    devices     = devices.lower()
    sel_markers = poc_markers.copy()
    if "vetscan" not in devices:
        sel_markers = [ m for m in sel_markers if m[0] != "v" ]
    if "istat" not in devices:
        sel_markers = [ m for m in sel_markers if m[0] != "i" ]
    if "icheck" not in devices:
        sel_markers = [ m for m in sel_markers if "otenoids" not in m ]
    if "all" in devices:
        sel_markers = poc_markers

    # add in the additional markers
    if type(add_features) is str:
        add_features = [add_features]
        
    if add_features[0].lower() == "none":
        features = sel_markers
    elif add_features[0].lower() == "all":
        # if you give pd.read_csv None as the "columns" param, it will read all columns
        return None
    else:
        features = sel_markers + add_features
        if "none" in features:
            features.remove("none")       
    return features

def write_train_test_data(X, y, save_bucket, save_prefix, label, data_type):
    out_df = pd.concat([X.reset_index(drop = True), y.reset_index(drop = True)], axis = 1)
    
    if save_bucket.lower() == "local":
        save_path = f"{save_prefix}/data/{label}"
        output_dir = Path(save_path)
        output_dir.mkdir(parents = True, exist_ok = True)
    else:
        save_path = f"s3://{save_bucket}/{save_prefix}/data/{label}"
    
    if data_type.lower() == "train":
        out_path = "/".join([save_path, "train_data.csv"])
    elif data_type.lower() == "test":
        out_path = "/".join([save_path, "test_data.csv"])
    else:
        raise ValueError(f"Unexpected value for data_type {data_type}, expected 'train' or 'test'")
        
    if save_path[0:2] == "s3":
        wr.s3.to_csv(df = out_df, path = out_path, index = False)
    else:
        out_df.to_csv(out_path, index = False)
                      
    return(f"Saved {data_type} data to {out_path}")
    
def write_pickle(bucket, key, out_obj):
    """  write the given python object to a given path in a given bucket or locally
    
    Arguments:
    bucket  -- a string specifying an s3 bucket name or the keyword "LOCAL"
    key     -- the path to write to
    out_obj -- the object to write 

    Returns:
    None
    """
    if bucket.upper() == "LOCAL":
        with open(key, 'wb') as f:
            pickle.dump(out_obj, f)
    else:
        pickle_byte_obj = pickle.dumps(out_obj)
        s3_resource     = boto3.resource('s3')
        s3_resource.Object(bucket,key).put(Body=pickle_byte_obj)