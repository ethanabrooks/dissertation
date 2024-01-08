from ray import tune

param_space = {
    "ppo_args/num_updates": tune.grid_search([4, 8]),
}
