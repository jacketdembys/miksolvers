import yaml
import os
import sys
import argparse




# Run the script for RMLP this way:
# python create_multi_experiments.py --blocks 1 --layers 1 --neurons 1024 --scale 10 --load local --seed 1 --jvar 1 --model rmlp  #--dataset ${dataset};

# set argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--layers",
                    type=int,
                    default=1,
                    help="Number of hidden layers.")
parser.add_argument("--neurons",
                    type=int,
                    default=100,
                    help="Number of neurons on each hidden layer.")
parser.add_argument("--scale",
                    type=int,
                    default=2,
                    help="Scale of the joints limits.")
parser.add_argument("--jvar",
                    type=int,
                    default=1,
                    help="Joint variation when generating dataset.")
parser.add_argument("--blocks",
                    type=int,
                    default=5,
                    help="Number of blocks if ResMLP or DenseMLP.")
parser.add_argument("--load",
                    type=str,
                    default='cloud',
                    help="local or cloud loading.")
parser.add_argument("--seed",
                    type=int,
                    default=1,
                    help="seed choice.")
parser.add_argument("--model",
                    type=str,
                    required=True,
                    help="model to run.")
parser.add_argument("--dataset",
                    type=str,
                    required=True,
                    help="dataset to run.")


if not len(sys.argv) > 1:
    #raise argparse.ArgumentError("Please, provide all arguments!")
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()


# important parameters
gpu_id = 0
layers = args.layers
neurons = args.neurons
scale = args.scale # 2 - 10
load_option = args.load
num_blocks = args.blocks
joint_variation = args.jvar
seed_choice = args.seed
model_str = args.model

if model_str == "rmlp":
    model = "ResMLP" 
elif model_str == "dmlp":
    model = "DenseMLP3" 
elif model_str == "mlp":
    model = "MLP"
elif model_str == "gpt2":
    model = "GPT2"
elif model_str == "gpt3":
    model = "GPT3"
elif model_str == "mdnmlp":
    model = "MDNMLP"
elif model_str == "rmdnmlp":
    model = "ResMDNMLP"

if args.dataset == "all6":
    robot_choice = 'All-6DoF'   #'7DoF-7R-Panda' '7DoF-GP66' 'All-6DoF' 'All-7DoF' 'All-DoFs' '3-to-10DoF'
    data_choice = 'combine-6DoF'
elif args.dataset == "all7":
    robot_choice = 'All-7DoF'   #'7DoF-7R-Panda' '7DoF-GP66' 'All-6DoF' 'All-7DoF' 'All-DoFs' '3-to-10DoF'
    data_choice = 'combine-7DoF'
elif args.dataset == "all67":
    robot_choice = 'All-67DoF'   #'7DoF-7R-Panda' '7DoF-GP66' 'All-6DoF' 'All-7DoF' 'All-DoFs' '3-to-10DoF'
    data_choice = 'combine-up-to-7DoF'
else:
    robot_choice = '7DoF-7R-Panda'   #'7DoF-7R-Panda' '7DoF-GP66' 
    data_choice = args.dataset

#print("\nDEBUG: {}".format(args.dataset))
#print("DEBUG: {}\n".format(robot_choice))

# read from path script
#for joint_variation in range(1,2):
#for scale in range(2,12,2):
#neuron = 1024
#for neuron in range(128, neurons+128, 128):
#for sr in range(args.seed):

#seed_choice = sr+1

# batch sizes: 4096, 65536
# build the content of the config file in a dictionary
config_info = {
    'NUM_EXPERIMENT_REPETITIONS': int(seed_choice),
    'ROBOT_CHOICE': robot_choice,
    'SEED_CHOICE': True,
    'SEED_NUMBER': int(seed_choice),
    'DEVICE_ID': int(gpu_id),
    'MODEL': {
        'NAME': model,      # MLP, ResMLP, DenseMLP3, DenseMLP, GPT2, GPT3
        'NUM_HIDDEN_LAYERS': layers,          
        'NUM_HIDDEN_NEURONS': neurons,
        'NUM_BLOCKS': num_blocks
    },             
    'TRAIN': {
        'DATASET': {
            'NUM_SAMPLES': 1000000,
            'JOINT_LIMIT_SCALE': int(scale),
            'JOINT_VARIATION': int(joint_variation),
            'TYPE': data_choice, # 1_to_1, seq, combine-6DoF, combine-7DoF, combine-up-to-7DoF, combine-up-to-10DoF
            'ORIENTATION': 'RPY' # RPY, Quaternion, DualQuaternion, Rotation, Rotation6d
        },
        'CHECKPOINT': {
            'SAVE_OPTIONS': 'cloud', # local, cloud
            'LOAD_OPTIONS': load_option,
            'PRETRAINED_G_MODEL': "",
            'RESUMED_G_MODEL': "",
        },
        'HYPERPARAMETERS': {
            'EPOCHS': 1000,
            'BATCH_SIZE': 128, #128, #100000
            'SHUFFLE': True,
            'NUM_WORKERS': 4,
            'PIN_MEMORY': False,
            'PERSISTENT_WORKERS': True,
            'OPTIMIZER_NAME': 'Adam', # Adam, SGD
            'LEARNING_RATE': 1e-4, #0.0001, # MLP / RMLP -> 0.001 and DMLP -> 0.0001
            'BETAS': [0.9, 0.999],
            'EPS': 0.00001,
            'WEIGHT_DECAY': 0.0,
            'WEIGHT_INITIALIZATION': 'default',
            'LOSS': 'mdn',           # lq, ld, mdn
        },
        'PRINT_EPOCHS': True,
        'PRINT_STEPS': 100
    },
}


#save_path = "configs/"+robot_choice+"/config_layers_"+str(int(layers))+"_neurons_"+str(int(neurons))+"_scale_"+str(int(scale))
#if not os.path.exists(save_path):
#            os.makedirs(save_path)

# open a yaml file and dump the content of the dictionary 
with open("train_seed_"+str(seed_choice)+".yaml", 'w') as yamlfile:
    data = yaml.dump(config_info, yamlfile)
    print("Successfully created the config file!")