import h5py
import pandas as pd
import json
import os
import numpy as np
import scipy

# examples of possible features
## Timeseries Features
class TimeFeatures:
    def __init__(self) -> None:
        pass
    #1 'mean'

    #2 'max'

    #3 'min'

    #4 rms
    def rms(x):
        #lambda x: np.sqrt(sum(x**2)/x.size)
        res = np.sqrt(np.mean(x**2))
        return res

    #5 abs_mean
    def abs_mean(x):
        res = np.mean(np.abs(x))
        return res

    #6 scipy.stats.kurtosis()
    def kurtosis(x):
        res = scipy.stats.kurtosis(x)
        return res

    #7 scipy.stats.skew()
    def skew(x):
        res = scipy.stats.skew(x)
        return res

    #8 std deviation

    #9 coefficient of variation
    def coef_var(x):
        res = scipy.stats.variation(x)
        return res

    #9 zero peak to peak: np.ptp (Barsczc2019)
    def zptp(x):
        res = np.ptp(x)/2
        return res

    #10 Crest Factor 
    def crest(x):
        res = (np.ptp(x)/2)/TimeFeatures.rms(x)
        return res

    #11 Impulse Factor (Ahmed2020/Wang2019b)
    def impulse_factor(x):
        res = x.max()/x.abs().mean()
        return res

    #12 Margin Factor (Ahmed2020)
    def margin_factor(x):
        peak= np.ptp(x)/2
        res = peak/((np.mean(np.sqrt(np.abs(x))))**2)
        return res

    #13 Shape Factor (Ahmed2020/Wang2019b)
    def shape_factor(x):
        rms = np.sqrt(np.mean(x**2))
        res = rms/(np.mean(np.abs(x)))
        return res

    #14 Clearance Factor (Ahmed2020/Wang2019b)
    def clearance_factor(x):
        res = x.max()/((np.mean(np.sqrt(np.abs(x))))**2)
        return res
    

def read_sensor(obj_features, data_paths):
    with h5py.File(data_paths, 'r') as hf:
        # Access the dataset and column names
        data = hf["data"]
        column_names = data.attrs["column_names"]
        
        # Pandas to read the data
        df = pd.DataFrame(data, columns=column_names)
    # add the word frontside/backside to not overwrite values
    file_name = data_paths.split('/')[-1]
    first_word = file_name.split('_')[0]

    for col in df.columns:
        if col in ["timestamp"]:
            continue
        # Calculate the mean of the column
        col_mean = df[col].mean()
        # Add the mean value to the dictionary with the column name as the key
        obj_features[first_word+"_"+col+"_"+"mean"] = col_mean
        
        # Calculate possible time series features
        obj_features[first_word+"_"+col+"_"+"rms"] = TimeFeatures.rms(df[col])
        obj_features[first_word+"_"+col+"_"+"abs_mean"] = TimeFeatures.abs_mean(df[col])
        obj_features[first_word+"_"+col+"_"+"kurtosis"] = TimeFeatures.kurtosis(df[col])
        obj_features[first_word+"_"+col+"_"+"skew"] = TimeFeatures.skew(df[col])
        obj_features[first_word+"_"+col+"_"+"coef_var"] = TimeFeatures.coef_var(df[col])
        obj_features[first_word+"_"+col+"_"+"zptp"] = TimeFeatures.zptp(df[col])
        obj_features[first_word+"_"+col+"_"+"crest"] = TimeFeatures.crest(df[col])
        obj_features[first_word+"_"+col+"_"+"impulse_factor"] = TimeFeatures.impulse_factor(df[col])
        obj_features[first_word+"_"+col+"_"+"margin_factor"] = TimeFeatures.margin_factor(df[col])
        obj_features[first_word+"_"+col+"_"+"shape_factor"] = TimeFeatures.shape_factor(df[col])
        obj_features[first_word+"_"+col+"_"+"clearance_factor"] = TimeFeatures.clearance_factor(df[col])

    return obj_features


def read_data():
    if os.path.exists('features.csv'):
        print('features exist, loading features')
        # Load the DataFrame from the saved CSV file
        file_path = 'features.csv';
        loaded_features = pd.read_csv(file_path)
        return loaded_features
    else:
        
        file_names = [
            'frontside_internal_machine_signals.h5',
            'frontside_external_sensor_signals.h5',
            'backside_external_sensor_signals.h5',
            'backside_internal_machine_signals.h5'
        ]
        
        meta_data = "data/cylinder_bottom_training/meta_data.json"
        
        # Load the json file
        with open(meta_data, "r") as f:
            data = json.load(f)
        
        # Initialize an empty dataframe to store the features
        features = pd.DataFrame()
                
        # Loop through each object in the data
        for obj in data:
            # Get the part_id and the anomaly
            anomaly = obj["process_data"][1]["anomaly"] # Assuming the anomaly is the same for all processes
            
            # Initialize a dictionary to store the features for this object
            obj_features = []
            obj_features = {"anomaly": int(anomaly)}
            
            # Loop through the process_data and load the data_paths
            for process_data in obj["process_data"]:
                for data_paths  in process_data["data_paths"]:
                    # if file path contain saw -> ignore
                    file_name = data_paths.split('/')[-1]
                    if file_name not in file_names:
                        continue
                    data_paths = "data/" + data_paths
                    data_paths = data_paths.replace("cylinder_bottom", "cylinder_bottom_training")
                    # Check the file extension
                    if data_paths.endswith(".h5"):
                        # # Load the h5 file
                        if os.path.exists(data_paths):
                            h5_file = h5py.File(data_paths, "r")
                            # # Do something with the h5 file, e.g. extract some statistics
                            obj_features = read_sensor(obj_features, data_paths)
                            # # Close the h5 file
                            h5_file.close()
                        else:
                            print("file does not exist: ")
                            print(data_paths)
                            
                    elif data_paths .endswith(".csv"):
                        data_paths = "data/" + data_paths
        
        
            # Append the obj_features dictionary to the features dataframe
            if features.empty:
                features = pd.DataFrame([obj_features])
            else:
                features = pd.concat([features, pd.DataFrame([obj_features])])
        
        # Specify the file path where you want to save the DataFrame
        file_path = 'features.csv'
        
        # Save the DataFrame to a CSV file
        features.to_csv(file_path, index=False)
        
        print(f'DataFrame saved to {file_path}')
        return features
