import h5py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import numpy as np

import random

#reads the data
def read_h52(file_path):
    if os.path.exists(file_path):
        try:
            with h5py.File(file_path, 'r') as h5_file:
                if "data" in h5_file:
                    data = h5_file["data"]
                    column_names = data.attrs["column_names"]

                    # Data of the first part
                    dataframe = pd.DataFrame(data, columns=column_names)
                    h5_file.close()

                    return dataframe
                else:
                    print("Key 'data' does not exist in the HDF5 file.")
                    print(file_path)
        except Exception as e:
            print(f"Error reading HDF5 file: {e}")
    else:
        print("File doesn't exist.")




# function builds mean of the timeline of features
def summation_function(n_sensor,q):
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
    
    a=q.index[0]  # the first sensor index for the first part

#     
    if n_sensor==0:
        sensor_name="/frontside_internal_machine_signals.h5"

    elif n_sensor==2:
        sensor_name="/frontside_external_sensor_signals.h5"

    elif n_sensor==5:
        sensor_name="/backside_internal_machine_signals.h5"

    elif n_sensor==4:
        sensor_name="/backside_external_sensor_signals.h5"

    else:
        print("not valid")
    
    
    data_paths = data[a]["process_data"][1]["data_paths"][n_sensor]
    #print(data_paths)
    part_id=data_paths.split("/")[3]
    data_paths = "data/cylinder_bottom_training/cnc_milling_machine/process_data/"+str(part_id)+str(sensor_name)
    #print(data_paths)
    #data_paths = data_paths.replace("cylinder_bottom", "cylinder_bottom_training")
    
    #print(data_paths)
    
    df_start=read_h52(data_paths)
   
    for j in range(1,len(q.index)):   # looping through the json file to get the data path for every part
        #print(j)
        a=q.index[j]
        #print(a)
   
        #print(b)
        data_paths = data[a]["process_data"][1]["data_paths"][n_sensor]
        #print(data_paths)
        part_id=data_paths.split("/")[3]
        data_paths = "data/cylinder_bottom_training/cnc_milling_machine/process_data/"+str(part_id)+str(sensor_name)
        #print(data_paths)
        #data_paths = data_paths.replace("cylinder_bottom", "cylinder_bottom_training")
    


        try:
            df = read_h52(data_paths)

        
            df_new=df_start.add(df)         # summing up the time series of the actual part with the series, that were summed before
            df_start=df_new
            df_end=df_start
        except: 
            print("couldn't evaluate")
            
    df_mean=df_end.div(len(q.index))         # dividing by the amount of the parts to obtain the mean
    return df_mean


def plot_function_3(df_plot):
    num_columns = len(df_plot.columns)
    num_rows = (num_columns + 3) // 4  # Ensure there are enough rows for all columns

    fig, axs = plt.subplots(num_rows, 4, figsize=(15, 3 * num_rows))

    # Flatten the axs array if there's only one row
    axs = axs.flatten()

    for i, element in enumerate(df_plot.columns):
        t = df_plot["timestamp"]
        y = df_plot[element]

        # Use axs[i] instead of plt.subplot
        axs[i].plot(t, y)
        axs[i].set_xlabel("t")
        axs[i].set_title(element)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
    
def plot_function(df_plot,sensors, title, top):
    num_columns = len(sensors)
    num_rows = (num_columns + 3) // 4  # Ensure there are enough rows for all columns

    fig, axs = plt.subplots(num_rows, 4, figsize=(15, 3 * num_rows))

    # Flatten the axs array if there's only one row
    axs = axs.flatten()

    for i, element in enumerate(df_plot[sensors]):
        t = df_plot["timestamp"]
        y = df_plot[element]

        # Use axs[i] instead of plt.subplot
        axs[i].plot(t, y)
        axs[i].set_xlabel("t")
        axs[i].set_title(element)

    # Adjust layout for better spacing
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, color='blue')
    # Adjust space between title and subplots
    plt.subplots_adjust(top=top)
    plt.show()


def plot_function_2(df_plot):
    j=1
    t= df_plot["timestamp"]

    fig = plt.figure(figsize=(40, 100))
    for element in df_plot.columns:
        y=df_plot[element]
        axs=plt.subplot(18,2,j)
        plt.plot(t,y)
        plt.xlabel("t")
        plt.title(element)
        j+=1

def plot_next(df_plot_1,df_plot_2,feature,side):
    num_columns = 2
    num_rows =1
    fig, axs = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

    # Flatten the axs array if there's only one row
    axs = axs.flatten()
    
    
    t_1= df_plot_1["timestamp"]
    t_2= df_plot_2["timestamp"]
    fig = plt.figure(figsize=(200, 200))
    y_1=df_plot_1[feature]
    y_2=df_plot_2[feature]
    
    # Use axs[i] instead of plt.subplot
    axs[0].plot(t_1,y_1)
    axs[0].set_xlabel("t")
    axs[0].set_title(str(side)+" "+str(feature)+" no anomaly")

    axs[1].plot(t_2,y_2)
    axs[1].set_xlabel("t")
    axs[1].set_title(str(side)+" "+str(feature)+" with anomaly")
    
    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()



def plot_histograms(a,b, selected_features):
    
    anomaly = selected_features.loc[selected_features['anomaly'] == 1]
    no_anomaly = selected_features.loc[selected_features['anomaly'] == 0]

    feature_number = 1
    c=b-a
    e=np.divide(c,4)
    d=int(np.ceil(e))
       
    fig, axs = plt.subplots(d, 4, figsize=[15, 5*d])  # One row, two columns
    
    feature_number = 1
    for param in selected_features.columns[a:b]:
        anomaly = selected_features.loc[selected_features['anomaly'] == 1]
        no_anomaly = selected_features.loc[selected_features['anomaly'] == 0]
        sbp = plt.subplot(d,4,feature_number)
        plt.hist(anomaly[param], alpha=0.5, label='with anomaly')
        plt.hist(no_anomaly[param], alpha=0.5, label='without anomaly')
        sbp.set_title(param)
        plt.tight_layout()
        plt.legend()
        feature_number+=1

    plt.show()

    

def plot_density(a,b, selected_features):
    feature_number = 1
    c=b-a
    e=np.divide(c,4)
    d=int(np.ceil(e))
    fig, axes = plt.subplots(d,4,figsize=[15,5*d])
    
    for param, ax in zip(selected_features.columns[a:b], axes.flatten()):
    
        sns.kdeplot(data=selected_features,x=param,hue="anomaly",fill=True,ax=ax)
        feature_number+=1
   
    plt.show()




def box_plot(a,b, selected_features):
    feature_number = 1
    c=b-a
    e=np.divide(c,4)
    d=int(np.ceil(e))
    fig, axs = plt.subplots(d, 4, figsize=[15, 5*d])  # One row, two columns

    
    for element in range(a,b):
        row_index=(element-a)//4
        col_index=(element-a)%4

        if d==1:
            selected_features.boxplot(column=selected_features.columns[element], by='anomaly',ax=axs[col_index])

        else:
            selected_features.boxplot(column=selected_features.columns[element], by='anomaly',ax=axs[row_index, col_index])
        feature_number+=1
    plt.show() 


def plot_random_anomaly_and_not(n_sensor,feature,q_with_anomaly,q_no_anomaly):
    meta_data = "data/cylinder_bottom_training/meta_data.json"
        
    # Load the json file
    with open(meta_data, "r") as f:
        data = json.load(f)
    
    

     
    if n_sensor==0:
        sensor_name="/frontside_internal_machine_signals.h5"

    elif n_sensor==2:
        sensor_name="/frontside_external_sensor_signals.h5"

    elif n_sensor==5:
        sensor_name="/backside_internal_machine_signals.h5"

    elif n_sensor==4:
        sensor_name="/backside_external_sensor_signals.h5"

    else:
        print("not valid")
    d=3
    fig, axs = plt.subplots(d, 2, figsize=[15, 5*d])  # One row, two columns


    for element in range(0,3):
     
    
        s=random.randint(0,len(q_with_anomaly))
        q=q_with_anomaly.index[s]
        data_paths = data[q]["process_data"][1]["data_paths"][n_sensor]
        #print(data_paths)
        part_id=data_paths.split("/")[3]
        data_paths = "/data/cylinder_bottom_training/cnc_milling_machine/process_data/"+str(part_id)+str(sensor_name)
        #print(data_paths)
        #data_paths = data_paths.replace("cylinder_bottom", "cylinder_bottom_training")
    
        #print(data_paths)
        part_with_anomaly=read_h52(data_paths)
        axs[element,1].plot(part_with_anomaly["timestamp"],part_with_anomaly[feature])
        axs[element,1].set_title(str(feature)+" with anomaly")
                
        
        s=random.randint(0,len(q_no_anomaly))
        q=q_no_anomaly.index[s]
        data_paths = data[q]["process_data"][1]["data_paths"][n_sensor]
        #print(data_paths)
        part_id=data_paths.split("/")[3]
        data_paths = "/data/cylinder_bottom_training/cnc_milling_machine/process_data/"+str(part_id)+str(sensor_name)
   
        #print(data_paths)
        part_no_anomaly=read_h52(data_paths)
        axs[element,0].plot(part_no_anomaly["timestamp"],part_no_anomaly[feature])
        axs[element,0].set_title(str(feature)+" no anomaly")

