python main.py                          \
--root '../../CANNON/'                   \
--dataset Coauthor-Phy                  \
--split_type random                     \
--seed 12345                            \
--learning_rate 0.001                   \
--num_input_dim 8415                    \
--num_hidden 1536                       \
--num_proj_hidden 1536                  \
--num_layers 2                          \
--num_proj_layers 2                     \
--drop_edge_rate_1 0.1                  \
--drop_feature_rate_1 0.4               \
--drop_edge_rate_2 0.4                  \
--drop_feature_rate_2 0.1               \
--tau 0.2                               \
--need_FA 'True'                         \
--FA_p 0.45                              \
--num_trials 5                          \
--num_epochs 1500                       \
--eval_freq 10                          \
--gpu_id 0                              \


