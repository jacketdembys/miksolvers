import wandb
run = wandb.init()
artifact = run.use_artifact('jacketdembys/ik-iros26/run-2b1gh6z5-history:v0', type='wandb-history')
artifact_dir = artifact.download()

print(artifact_dir)