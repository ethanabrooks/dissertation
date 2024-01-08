import wandb

api = wandb.Api()

group_name = "Sweep num_updates for 4x3-test"

for run_name in [
    "0sds7bw5",
    "4mb6zgqq",
    "i87w77ty",
    "ft6049mq",
]:
    run = api.run(f"rldl/icvi/{run_name}")
    for artifact in run.logged_artifacts():
        path = artifact.download()

runs = api.runs("rldl/icvi", {"$and": [{"group": group_name}]})
for run in runs:
    for artifact in run.logged_artifacts():
        path = artifact.download()
