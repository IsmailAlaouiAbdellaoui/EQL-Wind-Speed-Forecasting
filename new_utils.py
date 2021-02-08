import os
import re
import numpy as np
from matplotlib import pyplot as plt 
from sklearn.metrics import mean_squared_error, mean_absolute_error

def create_experiment_folder(experiment_number):
    try:
        path_new_experiment = "ExperimentsSR/Experiment" + str(experiment_number)
        os.mkdir(path_new_experiment)
    except Exception as e:
        print ("Creation of the directory {} failed".format(path_new_experiment))
        print("Exception error: ",str(e)) 

def get_experiment_number():
    experiments_folders_list = os.listdir(path='ExperimentsSR/')
    if(len(experiments_folders_list) == 0): #empty folder
        return 1
    else:  
        temp_numbers=[]
        for folder in experiments_folders_list:
            number = re.findall(r'\d+', folder)
            if(len(number)>0):
                temp_numbers.append(int(number[0]))
        return max(temp_numbers) + 1
    
def create_info_training_file(experiment_number):
    filename = "ExperimentsSR/Experiment"+str(experiment_number)+"/infos_experience"+str(experiment_number)+".txt"
    with open(filename, "w") as file:
        file.write("")
        
        
def create_main_experiment_folder():
    if(not os.path.isdir("ExperimentsSR")):
        try:
            os.mkdir("ExperimentsSR")
        except Exception as e:
            print ("Creation of the main experiment directory failed")
            print("Exception error: ",str(e))
            
            
def get_folders_started():
    create_main_experiment_folder()      
    experiment_number = get_experiment_number()
    create_experiment_folder(experiment_number)
    create_summary_file(experiment_number)
    return experiment_number

def record_base_info(experiment_number,**config):
    filename = "ExperimentsSR/Experiment{}/summary_experiment{}.txt".format(experiment_number,experiment_number)
    with open(filename, "a+") as file:
        for key,value in config.items():
            file.write("{}: {}\n".format(key,value))
        # file.write("use_rescaled_MSE: {}\n".format(config["use_rescaled_MSE"]))
        # file.write("epochs1: {}\n".format(config["epochs1"]))
        # file.write("epochs2: {}\n".format(config["epochs2"]))  
        # file.write("threshold_value: {}\n".format(config["threshold_value"]))
        # file.write("a_L_0.5: {}\n".format(config["a_L_0.5"]))
        # file.write("use_phase2: {}\n".format(config["use_phase2"]))
        # file.write("use_thresholding_before_phase2: {}\n".format(config["use_thresholding_before_phase2"]))
        # file.write("lambda_reg: {}\n".format(config["lambda_reg"]))
        # file.write("batch_size: {}\n".format(config["batch_size"]))
        # file.write("phase1_lr: {}\n".format(config["phase1_lr"]))
        # file.write("phase2_lr: {}\n".format(config["phase2_lr"]))
        # file.write("eql_number_layers: {}\n".format(config["eql_number_layers"]))
        # file.write("optimizer: {}\n".format(config["optimizer"]))
        # file.write("use_regularization_phase2: {}\n".format(config["use_regularization_phase2"]))
        # file.write("number_trials: {}\n".format(config["number_trials"]))
        # file.write("steps_ahead: {}\n".format(config["steps_ahead"]))
        # file.write("from_file: {}\n".format(config["from_file"]))
        # file.write("non_masked_weight_file: {}\n".format(config["non_masked_weight_file"]))
        
def create_summary_file(experiment_number):
    filename = "ExperimentsSR/Experiment{}/summary_experiment{}.txt".format(experiment_number,experiment_number)
    with open(filename, "w") as file:
        file.write("")
    # file = open(filename, "w+")
            
def plot_train_vs_validation(experiment_number,num_epochs,train_loss,validation_loss,phase):
    x = range(len(train_loss))
    plt.plot(x,train_loss,label="Training")
    plt.plot(x,validation_loss,label="Validation")
    plt.ylabel("MSE")
    plt.xlabel("Epochs")
    plt.title(phase)
    plt.legend()
    filename = "ExperimentsSR/Experiment{}/train_vs_validation.jpg".format(experiment_number)
    plt.savefig(filename)
    plt.clf()
    
def plot_histogram(experiment_number,weights,phase,type_weights,a):
    plt.hist(weights)
    plt.title(phase+", "+type_weights+", a:{}".format(a))
    filename = "ExperimentsSR/Experiment{}/hist_{}.jpg".format(experiment_number,type_weights)
    plt.savefig(filename)
    plt.clf()
    
    
def append_text_to_summary(experiment_number,text):
    filename = "ExperimentsSR/Experiment{}/summary_experiment{}.txt".format(experiment_number,experiment_number)
    with open(filename, "a+") as file:
        file.write(text)
        
def plot_descaled_real_vs_prediction(experiment_number,y_real,y_predicted,y_min,y_max):
    y_predicted_rescaled = y_predicted * (y_max - y_min) + y_min
    y_test_rescaled = y_real * (y_max - y_min) + y_min
    
    mse = mean_squared_error(y_predicted_rescaled,y_test_rescaled)
    mae = mean_absolute_error(y_predicted_rescaled,y_test_rescaled)
    
    plt.figure(figsize=(10,8))
    plt.plot(y_test_rescaled[0:2000],label="Real")
    plt.plot(y_predicted_rescaled[0:2000],label="Prediction from formula")
    plt.legend()
    plt.title("MAE: {:.2e}, MSE: {:.2e}".format(mae,mse))
    
    filename = "ExperimentsSR/Experiment{}/real_vs_prediction.jpg".format(experiment_number)
    plt.savefig(filename)
    
    plt.clf()
    return mae
    
def plot_descaled_real_vs_prediction_european(experiment_number,y_real,y_predicted,y_min,y_scale):
    y_predicted_rescaled = y_predicted - y_min
    y_predicted_rescaled /= y_scale
    
    y_test_rescaled = y_real  - y_min
    y_test_rescaled /= y_scale
    
    mse = mean_squared_error(y_predicted_rescaled,y_test_rescaled)
    mae = mean_absolute_error(y_predicted_rescaled,y_test_rescaled)
    
    plt.figure(figsize=(10,8))
    plt.plot(y_test_rescaled[0:2000],label="Real")
    plt.plot(y_predicted_rescaled[0:2000],label="Prediction from formula")
    plt.legend()
    plt.title("MAE: {:.2e}, MSE: {:.2e}".format(mae,mse))
    
    filename = "ExperimentsSR/Experiment{}/real_vs_prediction_phase2.jpg".format(experiment_number)
    plt.savefig(filename)
    
    plt.clf()
    
def get_sparsity_percent(array):
    num_zeros = np.count_nonzero(array==0)
    num_elements = array.size
    return num_zeros/num_elements*100


def get_size_network(weights):
    size_nn = 0
    for i in range(len(weights)):
        size_nn += weights[i].size
    return size_nn

def get_sparsity_nn(weights,size_nn):
    sparsity_nn = 0
    for i in range(len(weights)):
        num_elements = weights[i].size
        sparsity = get_sparsity_percent(weights[i])
        sparsity_nn += num_elements*sparsity/size_nn
    
    return sparsity_nn
        
    
def get_thresholds(weights):
    current_threshold = 0
    step_threshold = 1e-6
    thresholds = []
    offset = 1e-2
    boundary0 = (0,0+offset)
    boundary10 = (10-offset,10+offset)
    boundary20 = (20-offset,20+offset)
    boundary30 = (30-offset,30+offset)
    boundary40 = (40-offset,40+offset)
    boundary50 = (50-offset,50+offset)
    boundary60 = (60-offset,60+offset)
    boundary70 = (70-offset,70+offset)
    boundary80 = (80-offset,80+offset)
    boundary90 = (90-offset,90+offset)
    boundary100 = (100-offset,100)
    boundaries = [
        boundary0,
        boundary10,
        boundary20,
        boundary30,
        boundary40,
        boundary50,
        boundary60,
        boundary70,
        boundary80,
        boundary90,
        boundary100,
        ]
    size_nn = get_size_network(weights)
    for boundary in boundaries:
        print("processing boundary ",boundary)
        masked_weights = []
        count = 0
        while(True):
            masked_weights = []
            if count == 100:
                import sys
                sys.exit()
            # print("current threshold:",current_threshold)
            for item in weights:
                # mask = tf.cast(tf.constant(tf.abs(item) > kwargs["threshold"]),tf.float32)
                mask = (abs(item)>current_threshold)
                masked_weights.append(np.multiply(mask,item))
            sparsity_nn = get_sparsity_nn(masked_weights, size_nn)
            if sparsity_nn > 100:
                print("error, sparsity bigger than 100:")
            print("sparsity_nn: ",sparsity_nn)
            if sparsity_nn >= boundary[0] and sparsity_nn < boundary[1]:
                print("found right threshold: ",current_threshold)
                thresholds.append(current_threshold)
                break
            else:
                current_threshold += step_threshold
            count += 1
    return thresholds
        
    
    


