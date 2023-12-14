python main.py                          \
--root '../../CANNON/'                   \
--dataset DBLP                          \
--split_type random                     \
--seed 12345                            \
--learning_rate 0.0001                  \
--num_input_dim 1639                    \
--num_hidden 1536                        \
--num_proj_hidden 1536                   \
--num_layers 2                          \
--num_proj_layers 2                     \
--drop_edge_rate_1 0.1                  \
--drop_feature_rate_1 0.4               \
--drop_edge_rate_2 0.4                  \
--drop_feature_rate_2 0.1               \
--tau 0.7                               \
--need_FA False                         \
--FA_p 0.4                              \
--num_trials 5                          \
--num_epochs 1500                       \
--eval_freq 10                          \
--gpu_id 0                              \


