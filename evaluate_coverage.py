"""
Libraries
"""
#if DEBUG: print("Loading Libraries...")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import random
import sklearn
import time
import math
import matplotlib.pyplot as plt
import os
import sys
import wandb
import yaml
import argparse

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import tqdm
from scipy import stats

from utils import *
from models import *
from models_2 import DenseNet




import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def coverage_curve_mdn_anchors(
    model,
    iterator,
    device,
    robot_choice,
    S_list=[1, 2, 5, 10, 20],
    pos_thr_m=1e-3,                  # 1 mm in meters
    rot_thr_rad=np.deg2rad(1.0),     # 1 degree in radians
    use_norm_threshold=True          # True: ||pos|| and ||rot|| thresholds; False: per-axis thresholds
):
    """
    Computes Hit@S coverage curve for MDN anchors (mu) on a dataset iterator.

    model(x) must return (_, mdn_params) where mdn_params = (logits_pi, mu, log_sigma),
    and mu has shape [B, K, D].

    reconstruct_pose_modified(y_true_np, y_pred_np, robot_choice) must return (_, _, X_errors)
    where X_errors has shape [N, 6] = [dx,dy,dz, droll,dpitch,dyaw] in meters/radians.
    """

    model.eval()

    #S_list = list(S_list)
    maxS = max(S_list)

    total = 0
    hit_counts = np.zeros(len(S_list), dtype=np.int64)

    with torch.no_grad():
        for data in tqdm(iterator, desc="Coverage (MDN anchors)"):
            x = data["input"].to(device)     # [B, input_dim]
            y = data["output"].to(device)    # [B, D]

            # Forward through MDN
            _, mdn_params = model(x)
            logits_pi, mu, log_sigma = mdn_params    # mu: [B, K, D]
            B, K, D = mu.shape

            # Build candidate joints: (B*K, D)
            cand_y = mu.reshape(B * K, D)

            # Replicate y for FK comparison: (B*K, D)
            y_rep = y.unsqueeze(1).expand(B, K, D).reshape(B * K, D)

            # Move to numpy for your existing FK routine
            cand_y_np = cand_y.detach().cpu().numpy()
            y_rep_np = y_rep.detach().cpu().numpy()

            # Compute pose errors for all candidates: (B*K, 6)
            _, _, X_errors_all = reconstruct_pose_modified(y_rep_np, cand_y_np, robot_choice)

            # Reshape to (B, K, 6) then split into (B, K, 3) and (B, K, 3)
            X_err = X_errors_all.reshape(B, K, 6)
            pos_err = X_err[:, :, :3]    # (B, K, 3) meters
            rot_err = X_err[:, :, 3:]    # (B, K, 3) radians

            if use_norm_threshold:
                pos_norm = np.linalg.norm(pos_err, axis=2)  # (B, K)
                rot_norm = np.linalg.norm(rot_err, axis=2)  # (B, K)

                # valid[b,k] says candidate k solves target b within tolerances
                valid = (pos_norm <= pos_thr_m) & (rot_norm <= rot_thr_rad)  # (B, K)
            else:
                # per-axis thresholds (stricter / different behavior)
                valid_pos = (np.abs(pos_err) <= pos_thr_m).all(axis=2)  # (B, K)
                valid_rot = (np.abs(rot_err) <= rot_thr_rad).all(axis=2)  # (B, K)
                valid = valid_pos & valid_rot

            # Update Hit@S counts for each S
            # "first S anchors" means candidates 0..S-1
            for i, S in enumerate(S_list):
                hitS = valid[:, :S].any(axis=1)       # (B,) per target
                hit_counts[i] += int(hitS.sum())

            total += B

    hit_rates = hit_counts / max(total, 1)

    return np.array(S_list), hit_rates


def plot_coverage_curve(S_list, hit_rates, title="Coverage curve (Hit@S)"):
    plt.figure()
    plt.plot(S_list, hit_rates, marker="o")
    plt.ylim(0.0, 1.01)
    plt.xlabel("Number of candidate IK solutions (S)")
    plt.ylabel("Hit@S (fraction of targets with ≥1 valid solution)")
    plt.grid(True)
    plt.title(title)
    plt.show()


# ---------------------------
# Example usage (you plug in your objects):
# ---------------------------
# S_list, hit_rates = coverage_curve_mdn_anchors(
#     model=model,
#     iterator=test_loader,
#     device=device,
#     robot_choice=robot_choice,
#     reconstruct_pose_modified=reconstruct_pose_modified,
#     S_list=(1, 2, 5, 10, 20),
#     pos_thr_m=1e-3,
#     rot_thr_rad=np.deg2rad(1.0),
#     use_norm_threshold=True
# )
# plot_coverage_curve(S_list, hit_rates, title=f"MDN Anchors Coverage (K={model.K})")


if __name__ == '__main__':
    

    # Read parameters from configuration file
    print('Read from the config file...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path",
                        type=str,
                        default="train.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)
    
    # set parameters and configurations
    robot_choice = config["ROBOT_CHOICE"]
    seed_choice = config["SEED_CHOICE"]                                           # seed random generators for reproducibility
    seed_number = config["SEED_NUMBER"]
    print_epoch = config["TRAIN"]["PRINT_EPOCHS"]  
    batch_size = config["TRAIN"]["HYPERPARAMETERS"]["BATCH_SIZE"]                 # desired batch size
    init_type = config["TRAIN"]["HYPERPARAMETERS"]["WEIGHT_INITIALIZATION"]       # weights init method (default, uniform, normal, xavier_uniform, xavier_normal)
    #hidden_layer_sizes = [128,128,128,128]                                       # architecture to employ
    learning_rate = config["TRAIN"]["HYPERPARAMETERS"]["LEARNING_RATE"]           # learning rate
    optimizer_choice = config["TRAIN"]["HYPERPARAMETERS"]["OPTIMIZER_NAME"]       # optimizers (SGD, Adam, Adadelta, RMSprop)
    loss_choice =  config["TRAIN"]["HYPERPARAMETERS"]["LOSS"]                     # l2, l1, lfk
    K =  config["TRAIN"]["HYPERPARAMETERS"]["MDN_K"]
    network_type =  config["MODEL"]["NAME"] 
    num_blocks =  config["MODEL"]["NUM_BLOCKS"]     
    dataset_samples = config["TRAIN"]["DATASET"]["NUM_SAMPLES"]                   # MLP, ResMLP, DenseMLP, FouierMLP 
    print_steps = config["TRAIN"]["PRINT_STEPS"] 
    save_option = config["TRAIN"]["CHECKPOINT"]["SAVE_OPTIONS"]                                # local or cloud
    load_option = config["TRAIN"]["CHECKPOINT"]["LOAD_OPTIONS"]  
    dataset_type = config["TRAIN"]["DATASET"]["TYPE"]
    joint_steps = config["TRAIN"]["DATASET"]["JOINT_VARIATION"]
    orientation_type = config["TRAIN"]["DATASET"]["ORIENTATION"]

    scale = config["TRAIN"]["DATASET"]["JOINT_LIMIT_SCALE"]
    EPOCHS = config["TRAIN"]["HYPERPARAMETERS"]["EPOCHS"] 


    # Set other variables
    experiments = config["NUM_EXPERIMENT_REPETITIONS"]
    layers = config["MODEL"]["NUM_HIDDEN_LAYERS"]
    neurons = config["MODEL"]["NUM_HIDDEN_NEURONS"]   
    hidden_layer_sizes = np.zeros((1,layers))          
    hidden_layer_sizes[:,:] = neurons
    hidden_layer_sizes = hidden_layer_sizes.squeeze(0).astype(int).tolist()
    experiment_number = experiments

    # Use the right device       
    print("Check GPU device to run...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "Apple MPS GPU"
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{config['DEVICE_ID']}")
        device_name = torch.cuda.get_device_name(device)
    else:
        device = torch.device("cpu")
        device_name = "CPU"


    # Set up for th robot to run
    if robot_choice == "7DoF-7R-Panda" or robot_choice == "7DoF-GP66":
        if dataset_type == "1_to_1":
            n_DoF = 7
            output_dim = 7
            if orientation_type == "RPY":  
                input_dim = 6 #6 
            elif orientation_type == "Quaternion": 
                input_dim = 7 
            elif orientation_type == "DualQuaternion": 
                input_dim = 8 
            elif orientation_type == "Rotation": 
                input_dim = 12 
            elif orientation_type == "Rotation6d": 
                input_dim = 9 
        elif dataset_type == "seq":
            n_DoF = 7
            input_dim = 6+6+7 #6
            output_dim = 7

        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]

    # Get network architecture
    if network_type == "MLP":
        model = MLP(input_dim, hidden_layer_sizes, output_dim)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
        #model = MLP(mapping_size*2, hidden_layer_sizes, output_dim)
    elif network_type == "ResMLP":
        model = ResMLPSum(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
    elif network_type == "DenseMLP":
        model = DenseMLP(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
    elif network_type == "DenseMLP2":
        model = DenseMLP2(input_dim, neurons, output_dim, num_blocks)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
    elif network_type == "DenseMLP3":
        block_config = np.zeros((1,num_blocks))   
        block_config[:,:] = layers
        block_config = block_config.squeeze(0).astype(int).tolist()
        model = DenseNet(input_dim, neurons, block_config, output_dim)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
    elif network_type == "FourierMLP":
        fourier_dim = 16
        scale = 10
        model = FourierMLP(input_dim, fourier_dim, hidden_layer_sizes, output_dim, scale)
    elif network_type == "MDNMLP":
        #K = config.get("MDN_K", 100)
        model = MDNMLP(input_dim, hidden_layer_sizes, output_dim, K=K).to(device)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
    elif network_type == "ResMDNMLP":
        #K = config.get("MDN_K", 100)
        model = ResMDNMLPSum(input_dim, neurons, output_dim, num_blocks, K=K).to(device)
        save_layers_str = "blocks_"+ str(num_blocks)+"_layers_"+ str(layers)
        print(f"Using MDN with K = {model.K}")


    # Set loss function
    if loss_choice == "lq":
        criterion = nn.MSELoss(reduction="mean")
    elif loss_choice == "l1":
        criterion = nn.L1Loss(reduction="mean")
    elif loss_choice == "ld":
        criterion = FKLoss(robot_choice=robot_choice, device=device)
    elif loss_choice == "mdn":
        criterion = MDNLoss()


    
    # Seed for reproducibility
    if seed_choice:
        random.seed(seed_number)
        np.random.seed(seed_number)
        torch.manual_seed(seed_number)

    # Load test dataset 
    # Set up for th robot to run
    if robot_choice == "7DoF-7R-Panda" or robot_choice == "7DoF-GP66":
        if dataset_type == "1_to_1":
            n_DoF = 7
            output_dim = 7
            if orientation_type == "RPY":  
                input_dim = 6 #6 
            elif orientation_type == "Quaternion": 
                input_dim = 7 
            elif orientation_type == "DualQuaternion": 
                input_dim = 8 
            elif orientation_type == "Rotation": 
                input_dim = 12 
            elif orientation_type == "Rotation6d": 
                input_dim = 9 
        elif dataset_type == "seq":
            n_DoF = 7
            input_dim = 6+6+7 #6
            output_dim = 7

        pose_header = ["x", "y", "z","R","P","Y"]
        joint_header = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]


    
    print("Load dataset from local machine...")
    # Load dataset locally only
    if load_option == "local":
        if dataset_type == "1_to_1":
            filename = '../docker/datasets/7DoF-Combined/review_data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq_'+str(joint_steps)+'.csv'
            data = pd.read_csv(filename) #+'_'+orientation_type+'.csv')
        elif dataset_type == "seq":
            filename = '../docker/datasets/7DoF-Combined/review_data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq_'+str(joint_steps)+'.csv'
            data = pd.read_csv(filename)
    
    # Load dataset locally only
    elif load_option == "cloud":
        if dataset_type == "1_to_1":
            filename = '/home/datasets/7DoF-Combined/review_data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq_'+str(joint_steps)+'.csv'
            data = pd.read_csv(filename) #+'_'+orientation_type+'.csv')
        elif dataset_type == "seq":
            filename = '/home/datasets/7DoF-Combined/review_data_'+robot_choice+'_'+str(int(dataset_samples))+'_qlim_scale_'+str(int(scale))+'_seq_'+str(joint_steps)+'.csv'
            data = pd.read_csv(filename)
   
    data_a = np.array(data) 
    train_data_loader, test_data_loader, X_validate, y_validate, X_train, y_train, X_test, y_test = load_dataset(data_a, n_DoF, batch_size, robot_choice, dataset_type, device, input_dim)
    X_test = X_test[1000:1001,:]
    y_test = y_test[1000:1001,:]
    test_data_loader = load_test_dataset(X_test, y_test, device)
    
    
    # Load trained model
    weights_file = "./best_epoch_K"+str(K)+"_1000000.pth"
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
        
    # Evaluate
    
    with torch.no_grad():
        #results = inference_modified(model, test_data_loader, criterion, device, robot_choice)
        #results = inference_modified_best_of_k(model, test_data_loader, criterion, device, robot_choice)
        results = inference_modified_return_all_k(model, test_data_loader, criterion, device, robot_choice)


    print_inference_results(results, max_k_show=K)

    #print(results['X_errors'].shape)
    #print(results['X_errors'])
    
    """
    S_list, hit_rates = coverage_curve_mdn_anchors(
    model=model,
    iterator=test_data_loader,
    device=device,
    robot_choice=robot_choice,
    S_list=[5],
    pos_thr_m=1e-3,
    rot_thr_rad=np.deg2rad(1.0),
    use_norm_threshold=True
    )

    print(S_list)
    print(hit_rates)

    plot_coverage_curve(S_list, hit_rates, title=f"MDN Anchors Coverage (K={model.K})")
    """


