import wandb

api = wandb.Api()

ids = ["m1z0boc7", "404l91sm", "p1eilz3l"]

for id in ids:
    run = api.run(f"rldl/icvi/{id}")
    for artifact in run.logged_artifacts():
        path = artifact.download()
