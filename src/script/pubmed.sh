# python -m torch.distributed.launch --nproc_per_node 4   \     
# main.py                                 \
python main.py                          \
--root '../../CANNON/'                   \
--dataset PubMed                        \
--split_type random                     \
--seed 12345                            \
--learning_rate 0.0001                  \
--num_input_dim 500                     \
--num_hidden 896                        \
--num_proj_hidden 896                   \
--num_layers 2                          \
--num_proj_layers 2                     \
--drop_edge_rate_1 0.1                  \
--drop_feature_rate_1 0.4               \
--drop_edge_rate_2 0.4                  \
--drop_feature_rate_2 0.1               \
--tau 0.3                               \
--need_FA False                         \
--FA_p 0.5                              \
--num_trials 5                          \
--num_epochs 1500                       \
--eval_freq 10                          \
--gpu_id 0                              \

