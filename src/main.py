
import random
import torch
import numpy as np
import argparse

import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DataParallel as DP          # 


from model_CANNON import Runner

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',               type=str, default='../../CANNON/')
    parser.add_argument('--dataset',            type=str, default='Cora')
    parser.add_argument('--split_type',         type=str, default='random')
    
    parser.add_argument('--seed',               type=int, default=12345)
    parser.add_argument('--learning_rate',      type=float, default=0.0001)
    parser.add_argument('--num_input_dim',      type=int, default=1433)
    parser.add_argument('--num_hidden',         type=int, default=512)
    parser.add_argument('--num_proj_hidden',    type=int, default=512)
    parser.add_argument('--num_layers',         type=int, default=2)
    parser.add_argument('--num_proj_layers',    type=int, default=2)

    parser.add_argument('--drop_edge_rate_1',   type=float, default=0.2)
    parser.add_argument('--drop_feature_rate_1',type=float, default=0.3)
    parser.add_argument('--drop_edge_rate_2',   type=float, default=0.3)
    parser.add_argument('--drop_feature_rate_2',type=float, default=0.4)
    
    parser.add_argument('--tau',                type=float, default=0.4)
    parser.add_argument('--ratio',               type=float, default=0.5)     
    parser.add_argument('--lr_num_epochs',      type=int, default=5000)

    
    parser.add_argument('--need_FA',            type=str, default=False)
    parser.add_argument('--FA_p',               type=float, default=0.5)
    parser.add_argument('--num_trials',         type=int, default=10)
    parser.add_argument('--num_epochs',         type=int, default=800)
    parser.add_argument('--eval_freq',          type=int, default=10)                      
    parser.add_argument('--gpu_id',             type=int, default=0)

    parser.add_argument('--local_rank',         type=int, default=-1)          

    args = vars(parser.parse_args())
    args['data_dir'] = args['root'] + 'data'
    print('Dataset: {},       Model: CANNON'.format(args['dataset']))
    
    set_seed(args['seed'])    
    print('CANNON')
    
    #-------------------------------------------------------------------------------------------------------------------------#
    tot_res = []
    for ti in range(args['num_trials']):
        cur = Runner(conf=args).execute()
        print('==============================================================================')
        print('DATASET:{} | {}-th TRIAL | ACC: {}'.format(args['dataset'], ti, cur))
        print('==============================================================================')
    
        tot_res.append(cur)
    
    print(tot_res)
    res_mean = np.mean(tot_res)
    res_std = np.std(tot_res)
    print('The final reuslts of DATASET: {} | based on {} TRIALS | is ACC: {} +- {}'.format(args['dataset'], args['num_trials'], res_mean, res_std))
    
    