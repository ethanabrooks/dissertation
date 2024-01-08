import wandb

api = wandb.Api()

group_name = "Sweep num_updates for 2:1 and CQL"

runs = api.runs("rldl/icvi", {"$and": [{"group": group_name}]})
for run in runs:
    for artifact in run.logged_artifacts():
        path = artifact.download()
