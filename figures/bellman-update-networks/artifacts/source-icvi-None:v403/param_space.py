from ray import tune

param_space = {
    "ppo_args/num_updates": tune.grid_search([4, 6, 8]),
    # "train_args/policy_improvement_values": tune.grid_search([True, False]),
    # "model_args/conservative_loss_coef": tune.sample_from(
    #     lambda s: 0.1 if s.config["train_args/policy_improvement_values"] else None
    # ),
}
