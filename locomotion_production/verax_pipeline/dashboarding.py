import pandas as pd
import boto3
import pickle
from io import BytesIO
import holoviews as hv

def get_performance_tables(report_dict):
    report_df = pd.DataFrame.from_dict(report_dict).T.rename_axis('Class')
    report_df.index = pd.Series(report_df.index).replace({'0' : "negative", '1' : 'positive'})
    
    acc = report_df.loc["accuracy"]
    acc_val = acc.iloc[0].round(3)
    acc_df  = pd.DataFrame({"Class" : ["", "ACCURACY:"], "f1-score" : ["", acc_val], 
                      "precision" : [""] * 2, "recall" : [""] * 2, "support" : [""] *2})
    
    report_df = report_df.drop("accuracy").reset_index().round(3)
    return hv.Table(report_df), acc_val

def get_hist_df(reports, archive_bucket, report_dir, vetscore):
    idx_list  = []
    metric_dict = {}
    
    for rep_path in reports:
        report_dict = load_s3_pickle(archive_bucket, f"{report_dir}/{rep_path}")
        try:
            vetscore_dict = report_dict[vetscore]["1.0"]
        except KeyError:
            continue

        date = rep_path.split("_reports")[0].replace("_", "-")
        idx_list.append(date)

        vetscore_dict["accuracy"]        = report_dict[vetscore]["accuracy"]
        vetscore_dict["test_samples"]    = report_dict[vetscore]["macro avg"]["support"]
        vetscore_dict["train_positives"] = report_dict[vetscore]["train_data"]["1.0"]
        vetscore_dict["train_samples"]   = report_dict[vetscore]["train_data"]["0.0"] + vetscore_dict["train_positives"]
        for key in vetscore_dict:
            if key in metric_dict:
                metric_dict[key].append(vetscore_dict[key])
            else:
                metric_dict[key] = [vetscore_dict[key]]
    
    hist_df = pd.DataFrame(metric_dict, index = idx_list)
    hist_df = hist_df.reset_index().rename({"index" : "date", "f1-score" : "f1"}, axis = 1)
    hist_df = hist_df.sort_values(by = "date")
    
    return hist_df

def load_s3_pickle(bucket, filepath):
    s3 = boto3.resource('s3')
    with BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(filepath, data)
        data.seek(0)    # move back to the beginning after writing
        loaded_pickle = pickle.load(data)
    return loaded_pickle
