"""
import wandb
run = wandb.init()
artifact = run.use_artifact('jacketdembys/ik-iros26/run-2b1gh6z5-history:v0', type='wandb-history')
artifact_dir = artifact.download()

print(artifact_dir)
"""

# Download the artifact
import wandb
run = wandb.init(project="ik-iros26")
artifact = run.use_artifact(
    "jacketdembys/ik-iros26/inference_summary_ResMDNMLP_7DoF-7R-Panda_1:v0",
    type="results"
)
artifact_dir = artifact.download()
print("Artifact downloaded to:", artifact_dir)


# Access the artifact
import os
print("Files in artifact:")
for root, dirs, files in os.walk(artifact_dir):
    for f in files:
        print(os.path.join(root, f))


# Access the artifact file 
import pandas as pd
import os
csv_path = os.path.join(
    artifact_dir,
    "inference_summary_ResMDNMLP_7DoF-7R-Panda_1.csv"
)
df = pd.read_csv(csv_path)
print(df)


import glob
csv_files = glob.glob(os.path.join(artifact_dir, "*.csv"))
print(csv_files)
df = pd.read_csv(csv_files[0])
print(df)