import new_utils as nutils
import h5py
import tensorflow as tf
from utils import functions, pretty_print
import numpy as np
from utils.symbolic_network import SymbolicNet
from scipy.io import loadmat
from tqdm import tqdm
from utils.regularization import l12_smooth
from collections import OrderedDict
from time import sleep
import sys
from pytexit import py2tex
import time
import copy 

x_dim = 80
init_sd_first = 0.1
init_sd_middle = 0.5
init_sd_last = 1.0

batch_size = 200

activation_funcs = [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            *[functions.Sin()] * 2,
            *[functions.Sigmoid()] * 2,
            *[functions.Product()] * 2
        ]
n_layers = 2
n_double = functions.count_double(activation_funcs)
width = len(activation_funcs)

steps_ahead_to_step = {6:"1",
                       12:"2",
                       18:"3",
                       24:"4"}

alphabet_size = 26
num_variables = 80

def get_latex_equation_from_string(string):
    return py2tex(string)
    

def generate_variable_list(alphabet_size,num_variables):
    lower_case = []
    for i in range(26):
        lower_case.append(chr(i+97))
    
    variables = lower_case[:alphabet_size]
    prefix_index = 0
    letter_index = 0
    prefix = variables[0]
    for i in range(num_variables):
        if ((i+1)) % alphabet_size == 0:   
            variables.append(prefix+lower_case[letter_index])
            prefix_index += 1
            prefix = variables[prefix_index]
            letter_index =0
            
        else:
            variables.append(prefix+lower_case[letter_index])
            letter_index +=1
            
    return variables

var_names = generate_variable_list(alphabet_size,num_variables)

dict_vars = {}
for i in range(len(var_names)):
    dict_vars[var_names[i]] = i

def tensor_to_matrix(tensor): 
    number_features = tensor.shape[0]
    number_cities = tensor.shape[1]
    len_dates = tensor.shape[2]
    
    matrix_for_scaling = np.ones((len_dates,number_cities*number_features))
    for i in range(number_cities):
        for j in range(len_dates):
            features_city = tensor[:,i,j]
            matrix_for_scaling[j,i*number_features:(i+1)*number_features] = features_city        
    return matrix_for_scaling


def get_x_right_format(phase,steps_ahead,feature):
    filename = "{}/step{}.mat".format(feature,steps_ahead_to_step[steps_ahead])   
    data = loadmat('Denmark_data/{}'.format(filename))
    if phase == "train":
        x = data["Xtr"]
    elif phase == "test":
        x = data["Xtest"] 
    x = np.transpose(x,(0,3,2,1)) # => (Dates,Features,Lags,Cities)
    x = np.transpose(x,(0,1,3,2)) # => (Dates,Features,Cities,Lags)
    
    all_features_all_cities = x.shape[1] * x.shape[2]
    x_output = np.zeros((x.shape[0],80))
    for i in range(x.shape[0]):
        temp_matrix = tensor_to_matrix(x[i])
        for j in range(temp_matrix.shape[0]):#iterate through lags of matrix
            x_output[i,j*all_features_all_cities:(j+1)*all_features_all_cities] = temp_matrix[j]
    return x_output


def load_weights(filename):
    weights = []
    with h5py.File(filename, "r") as f:
        for i in range(3):
            weights.append(f.get('dataset{}'.format(i))[()])
    return weights

def generate_all_data(phase,steps_ahead,feature,city_index):
    filename = "{}/step{}.mat".format(feature,steps_ahead_to_step[steps_ahead]) 
    data = loadmat('Denmark_data/{}'.format(filename))
    if phase == "train":
        y_temp = data["Ytr"][:,city_index]
    elif phase == "test":
        y_temp = data["Ytest"][:,city_index]
    else:
        return
        
    x = get_x_right_format(phase,steps_ahead,feature).astype("float32")    
    y = y_temp.reshape(y_temp.shape[0],1).astype('float32')
    
    return x,y

def custom_loss(model, x, y_real, use_rescaled,reg_weight,a):
    y_predicted = model(x)
    if use_rescaled:
        y_predicted_rescaled = y_predicted * (y_max - y_min) + y_min
        y_real_rescaled = y_real * (y_max - y_min) + y_min
        error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y_real_rescaled, y_predicted_rescaled)#double check
    else:
        error = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y_real, y_predicted)
    reg_loss = l12_smooth(model.get_weights(),a)
    loss = error + reg_weight * reg_loss
    return loss

def grad(model, inputs, targets,use_rescaled_loss,reg_weight,a):
    with tf.GradientTape() as tape:
        loss_value = custom_loss(model, inputs, targets,use_rescaled_loss,reg_weight,a)
    return loss_value, tape.gradient(loss_value, model.get_weights())

def save_weights(weights,experiment_number,phase,best_val_loss):
    if phase == "phase1":
        filename_output = "ExperimentsSR/Experiment{}/nt_val_weights_{:.2e}.hdf5".format(experiment_number,best_val_loss)
    elif phase == "phase2":
        filename_output = "ExperimentsSR/Experiment{}/phase2_val_weights_{:.2e}.hdf5".format(experiment_number,best_val_loss)
    with h5py.File(filename_output, "w") as f:
        for i in range(len(weights)):
            f.create_dataset('dataset{}'.format(i), data=weights[i].numpy())
            

def mask_weights(masks,weights,destination):
    masked_weights = []
    for mask,weight in (zip(masks,weights)):
        
        if destination == "inside_nn":
            masked_weight = np.multiply(mask,weight.numpy())
            var_masked_weight = tf.Variable(masked_weight)
            masked_weights.append(var_masked_weight)
        elif destination == "input_nn":
            masked_weight = np.multiply(mask,weight)
            masked_weights.append(masked_weight)
    return masked_weights

def get_sparsity_eql(weights_eql,size_nn):
    weights_numpy = []
    for item in weights_eql:
        weights_numpy.append(item.numpy())
    sparsity_nn = 0
    for i in range(len(weights_numpy)):
        num_elements = weights_numpy[i].size
        sparsity = nutils.get_sparsity_percent(weights_numpy[i])
        sparsity_nn += num_elements*sparsity/size_nn
    return sparsity_nn

def train_masked(epochs_second_phase,steps_ahead,feature,city_index,**kwargs):
    count_loss_stagnation = 0
    weights = load_weights(kwargs["filename"])
    masks = []
    current_threshold = kwargs["threshold_value"]

    for item in weights:
        mask = (abs(item)>current_threshold)
        masks.append(mask)

    masked_weights = mask_weights(masks, weights, "input_nn")
    sparsity1 = nutils.get_sparsity_percent(masked_weights[0])
    sparsity2 = nutils.get_sparsity_percent(masked_weights[1])
    sparsity3 = nutils.get_sparsity_percent(masked_weights[2])
    size_nn = nutils.get_size_network(masked_weights)
    sparsity_nn_beg = nutils.get_sparsity_nn(masked_weights,size_nn)

    nutils.append_text_to_summary(kwargs["experiment_number"],"sparsity1 phase2 after masking: {}\n".format(sparsity1))
    nutils.append_text_to_summary(kwargs["experiment_number"],"sparsity2 phase2 after masking: {}\n".format(sparsity2))
    nutils.append_text_to_summary(kwargs["experiment_number"],"sparsity3 phase2 after masking: {}\n".format(sparsity3))
    nutils.append_text_to_summary(kwargs["experiment_number"],"sparsity_nn: {}\n".format(sparsity_nn_beg))
    
    masked_model = SymbolicNet(n_layers,
                              funcs=activation_funcs,
                              initial_weights=masked_weights)
    
    
    train_op = tf.keras.optimizers.RMSprop(learning_rate=kwargs["phase2_lr"])
    train_loss_results = []
    valid_loss_results = []
    x_train,y_train = generate_all_data("train",steps_ahead,feature,city_index)
    x_test,y_test = generate_all_data("test",steps_ahead,feature,city_index)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)
    
    best_val_loss = float('inf')
    best_val_model = masked_model
    best_val_weights = None
    
    
    for epoch in range(epochs_second_phase):
        epoch_loss_avg = tf.keras.metrics.MeanSquaredError()

        with tqdm(train_dataset,position = 0, leave = True,colour="#cc9933") as bar_train:
            for x,y in bar_train:               
                bar_train.set_description("Epoch {}".format(epoch+1))
                y_prediction = masked_model(x)
                loss_value, grads = grad(masked_model, x, y,kwargs["use_rescaled_loss"],kwargs["reg_weight"],kwargs["a"])
                train_op.apply_gradients(zip(grads, masked_model.get_weights()))
                
                network_weights = masked_model.get_weights()
                masked_weights = mask_weights(masks,network_weights,"inside_nn")
                masked_model.set_weights(masked_weights)
                
                epoch_loss_avg.update_state(y,y_prediction)  # Add current batch loss
                od = OrderedDict() 
                od["loss"] = f'{loss_value:.2f}'
                od["mse"] = f'{epoch_loss_avg.result():.2e}'
                bar_train.set_postfix(od)
                bar_train.update()
                if np.isnan(loss_value) or np.isnan(epoch_loss_avg.result()):
                    sys.exit("Nan value, stopping")
                sleep(0.05)

        train_loss_results.append(epoch_loss_avg.result())

        print("\n\nValidation\n")
        epoch_loss_avg_val = tf.keras.metrics.MeanSquaredError()
        with tqdm(val_dataset,position = 0, leave = True,colour="#cc3333") as bar_val:
            for x_val,y_val in bar_val:
                sleep(0.05)
                y_prediction_val = masked_model(x_val)
                loss_value_val, grads = grad(masked_model, x_val, y_val,kwargs["use_rescaled_loss"],kwargs["reg_weight"],kwargs["a"])
                
                
                epoch_loss_avg_val.update_state(y_val,y_prediction_val)
                if np.isnan(loss_value_val) or np.isnan(epoch_loss_avg_val.result()):
                    sys.exit("Nan value, stopping")
                
                od_val = OrderedDict() 
                od_val["loss_val"] = f'{loss_value_val:.2f}'
                od_val["mse_val"] = f'{epoch_loss_avg_val.result():.2e}'
                bar_val.set_description("Val Epoch {}".format(epoch+1))
                bar_val.set_postfix(od_val)
            if (epoch_loss_avg_val.result() < best_val_loss) and (epoch_loss_avg_val.result() != best_val_loss):
                print("\n==> Validation loss decreased from {:.2e} to {:.2e}".format(best_val_loss,epoch_loss_avg_val.result()))
                print("==> Saving new masked model")
                best_val_loss = epoch_loss_avg_val.result()
                best_val_model = copy.deepcopy(masked_model)
                best_val_weights = copy.deepcopy(masked_model.get_weights())
                
                sparsity_nn_end2 = get_sparsity_eql(best_val_weights,size_nn)
                print(sparsity_nn_end2)
                
                count_loss_stagnation = 0
            else:
                count_loss_stagnation += 1
        
        valid_loss_results.append(epoch_loss_avg_val.result())     
        print("\n\n"+"-"*6+" End of Epoch "+"-"*6+"\n\n")
        if count_loss_stagnation >= 4 and epoch > 50:
            print("No improvement over more than 4 epochs, and more than 50 epochs passed\nExiting phase 2...")
            break
        
        
    nutils.append_text_to_summary(kwargs["experiment_number"],"phase 2 best MSE validation: {}\n".format(best_val_loss))
    
        
    final_weights_list = []
    for item in best_val_weights:
        final_weights_list.append(item.numpy())
        
    sparsity_nn_end = nutils.get_sparsity_nn(final_weights_list,size_nn)
    
    assert sparsity_nn_beg == sparsity_nn_end, sys.exit("Sparsities not equal: beg = {}, end = {} ".format(sparsity_nn_beg,sparsity_nn_end))
        
    expr = pretty_print.network(final_weights_list, activation_funcs, var_names[:x_dim],threshold = 0)
    print("Formula from pretty print:",expr)
    nutils.append_text_to_summary(kwargs["experiment_number"],"Formula after phase2: {}\n".format(expr))
    try:
        latex = get_latex_equation_from_string(str(expr))
        clean_latex = latex.replace(r"\right","").replace(r"\left","").replace("$$","")
        nutils.append_text_to_summary(kwargs["experiment_number"],"Latex formula after phase2: {}\n".format(clean_latex))
    except Exception as e:
        print("\n\nCouldn't transform the string into latex format")
        print("=>Reason: ",e)
    
    save_weights(best_val_weights,kwargs["experiment_number"],"phase2",best_val_loss)
    
    nutils.plot_train_vs_validation(kwargs["experiment_number"], epochs_second_phase, train_loss_results, valid_loss_results,"phase2")
              
    y_predicted = best_val_model(x_test)
    mae = nutils.plot_descaled_real_vs_prediction(kwargs["experiment_number"],y_test,y_predicted,y_min,y_max)
    nutils.append_text_to_summary(kwargs["experiment_number"],"MAE: {}\n".format(mae))
    print("MAE: {}\n".format(mae))
    return best_val_model,best_val_weights

target_cities = ["Esbjerg", "Odense", "Roskilde"]

config = {"masked_training":True,
          "use_rescaled_loss":True,
          "step_ahead":6,
          "weather_feature":"wind_speed",#wind_speed or temp
          "sparsity":98,
           "filename":r"ExperimentsSR\Experiment118\nt_val_weights_6.35e-03.hdf5",
          
          "target_city": "Esbjerg", # Esbjerg or Odense or Roskilde
          
          "epochs2":100,
          "a":5e-4, #decreasing = more low weight values
          "reg_weight":0, #increasing = more low weight values
          
          "batch_size":200,
          "phase2_lr":1e-5,
          "eql_number_layers":2,
          "optimizer":"rmsprop",
          "number_trials":1
          
          }

sparsity_to_threshold = {
    98:7.5e-3
    }



if __name__ == "__main__":
    sparsity = config["sparsity"]
    config["threshold_value"] = sparsity_to_threshold[sparsity]
    
    target_city = config["target_city"]
    target_city_index = target_cities.index(target_city)
    
    steps_ahead = config["step_ahead"]
    feature = config["weather_feature"]
    filename = "{}/scale{}.mat".format(feature,steps_ahead_to_step[steps_ahead])
    data = loadmat('Denmark_data/{}'.format(filename))
    
    y_min = data["y_min_tr"][0][target_city_index] #double check !
    y_max = data["y_max_tr"][0][target_city_index] #double check !
    optimizer = config["optimizer"]
    second_phase_lr = config["phase2_lr"]
    number_epochs2 = config["epochs2"]
    
    
    if optimizer == "rmsprop":
        phase2_optimizer = tf.keras.optimizers.RMSprop(learning_rate=second_phase_lr)
    elif optimizer == "adam":
        phase2_optimizer = tf.keras.optimizers.Adam(learning_rate=second_phase_lr)
    else:
        sys.exit("Invalid optimizer, exiting ...")
    
    experiment_number = nutils.get_folders_started()
    
    print("\n\n"+"*"*10+" Beginning of Experiment {} ".format(experiment_number)+"*"*10+"\n\n")
    print("\n"+"-"*5+" Feature : {} ".format(feature)+"-"*5)
    print("\n"+"-"*5+" Steps ahead : {} ".format(steps_ahead)+"-"*5)
    print("\nSparsity of {} % using a threshold of {}\n\n".format(sparsity,config["threshold_value"]))
    
    nutils.record_base_info(experiment_number,**config)
    start = time.time()
    
    config["experiment_number"] = experiment_number
    m_model,m_weights =  train_masked(number_epochs2,steps_ahead,feature,target_city_index,**config)
    end = time.time()
    span = end - start
    print("Time span of second phase: {}".format(span))
    nutils.append_text_to_summary(experiment_number,"Time span of second phase: {}".format(span))
    print("\n\n"+"*"*10+" End of Experiment {} ".format(experiment_number)+"*"*10+"\n\n")
    print("\n"+"-"*5+" Feature : {} ".format(feature)+"-"*5)
    print("\n"+"-"*5+" Steps ahead : {} ".format(steps_ahead)+"-"*5)
    print("\nSparsity of {}% using a threshold of {}\n\n".format(sparsity,config["threshold_value"]))
    

