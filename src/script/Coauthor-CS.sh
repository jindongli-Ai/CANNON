python main.py                          \
--root '../../CANNON/'                   \
--dataset Coauthor-CS                   \
--split_type random                     \
--seed 12345                            \
--learning_rate 0.00005                   \
--num_input_dim 6805                    \
--num_hidden 1536                        \
--num_proj_hidden 1536                  \
--num_layers 2                          \
--num_proj_layers 2                     \
--drop_edge_rate_1 0.0                  \
--drop_feature_rate_1 0.7               \
--drop_edge_rate_2 0.5                  \
--drop_feature_rate_2 0.2               \
--tau 0.4                               \
--need_FA False                         \
--FA_p 0.5                              \
--num_trials 5                          \
--num_epochs 1500                       \
--eval_freq 10                          \
--gpu_id 0                              \


