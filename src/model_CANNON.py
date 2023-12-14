
# import wandb
import numpy as np
# from torch_geometric.nn.models import LightGCN
from base import BaseGSSLRunner
import torch
import torch.nn.functional as F
import GCL.augmentors as A
# from GCL.eval import get_split, from_predefined_split
from eval import get_split      ##
from eval import LREvaluator    ##
from tqdm import tqdm
from util.helper import _similarity
from torch.optim import Adam
from util.data import get_dataset
from util.utils import convert_to_dgl_graph, generate_rwr_subgraph
import torch.nn as nn

from torch_geometric.nn import GCNConv
# from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import reset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling

import sys
import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP    #
# from torch.nn.parallel import DataParallel as DP                #

# from plot_TSNE import tsne_plot_2D_x01_x02_x1_x2, tsne_plot_2D_x1_x2_Fused, plot_TSNE_x_y                #### 

class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layers):
        super(GCNEncoder, self).__init__()
        self.activation = activation()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim, cached=False))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, cached=False))
    
    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for i, conv in enumerate(self.layers):
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss, mode, intraview_negs=False, device='cpu', **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.device = device
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)

        # wandb.log({'sim_diff': sim_diff.item() / h1.size(0)})
        return (l1 + l2) * 0.5


class InfoNCE(object):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau
    
    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss


class CANNON(torch.nn.Module):
    def __init__(self, encoders, augmentor, hidden_dim, proj_dim, device):
        super(CANNON, self).__init__()
        self.subgraph_size = 5
        self.GNN_1, self.GNN_2 = encoders        
        self.augmentor = augmentor
        self.device = device
        self.hidden_dim = hidden_dim
        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)
        self.TCF = TCF(hidden_dim, hidden_dim, device)


    def forward(self, x, edge_index, edge_weight=None):

        aug1, aug2 = self.augmentor
        x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
        x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
        
        z01 = self.GNN_1(x, edge_index, edge_weight=edge_weight)        
        z1 = self.GNN_1(x1, edge_index1, edge_weight=edge_weight1)      

        z02 = self.GNN_2(x, edge_index, edge_weight=edge_weight)        
        z2 = self.GNN_2(x2, edge_index2, edge_weight=edge_weight2)      
        
        return z01, z02, z1, z2

    ################################### Projection Head ####################################

    def project_1(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def project_2(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)


class Runner(BaseGSSLRunner):
    def __init__(self, conf, **kwargs):
        super().__init__(conf, **kwargs)
    
    def load_dataset(self):
        self.dataset = get_dataset(self.config['data_dir'], self.config['dataset'])

        self.data = self.dataset[0].to(self.device)
        
    def train(self):

        aug1 = A.Compose([A.FeatureMasking(pf=self.config['drop_feature_rate_1']), 
                          A.EdgeRemoving(pe=self.config['drop_edge_rate_1'])])
        aug2 = A.Compose([A.FeatureMasking(pf=self.config['drop_feature_rate_2']),
                          A.EdgeRemoving(pe=self.config['drop_edge_rate_2'])])
        
        GCN_1 = GCNEncoder(input_dim=self.dataset.num_features,
                    hidden_dim=self.config['num_hidden'], activation=torch.nn.ReLU, num_layers=self.config['num_layers']).to(self.device)
        
        GCN_2 = GCNEncoder(input_dim=self.dataset.num_features,
                    hidden_dim=self.config['num_hidden'], activation=torch.nn.ReLU, num_layers=self.config['num_layers']).to(self.device)
        

        self.model = CANNON(encoders=(GCN_1, GCN_2), 
                            augmentor=(aug1, aug2),
                            hidden_dim=self.config['num_hidden'],
                            proj_dim=self.config['num_proj_hidden'],
                            device=self.device)
        self.model = self.model.to(self.device)
        # self.model = DP(self.model, device_ids=[2, 3])        #
        # self.model = DDP(self.model, device_ids=[self.config['local_rank']], output_device=self.config['local_rank'])   #

        contrast_model = DualBranchContrast(loss=InfoNCE(
            tau=self.config['tau']), mode='L2L', intraview_negs=True, device=self.device).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])
        

        tot_res = []
        cur_best_F1mi = 0
        cur_best_test_ACC = 0
        

        with tqdm(total=self.config['num_epochs'], desc='(T)') as pbar:
            for epoch in range(1, self.config['num_epochs'] + 1):
                self.model.train()
                optimizer.zero_grad()

                z01, z02, z1, z2 = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
          
                #------------------------------- FA start ------------------------------#

                if self.config['need_FA'] == 'True':
                    # print(self.config['need_FA'])
                    # print(self.config['FA_p'])
                    
                    k = torch.tensor(int(z01.shape[0] * self.config['FA_p']))
                    p = (1/torch.sqrt(k))*torch.randn(k, z01.shape[0]).to(self.device)

                    z01 = p @ z01
                    z02 = p @ z02 
                    z1 = p @ z1
                    z2 = p @ z2
                #-------------------------------- FA end -----------------------------#


                h01 = self.model.project_1(z01)
                h1 = self.model.project_1(z1)
                
                h02 = self.model.project_2(z02)
                h2 = self.model.project_2(z2)
                
                h0 = self.model.TCF(h01, h02)
                hc = self.model.TCF(h1, h2)

                combined = None
     
                L1 = contrast_model(h1, h2)
                L2 = contrast_model(h01, h2) + contrast_model(h02, h1)
                L3 = contrast_model(h0, hc)

                loss = L1 + L2/2 + L3/2

                
                loss.backward()
                optimizer.step()

                # wandb.log({'loss': loss.item()})
                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

                if epoch % self.config['eval_freq'] == 0:
                    cur_F1mi, cur_val_acc, cur_test_acc = self.test(self.config['dataset'], epoch=epoch, t=self.config["split_type"])
                    cur_best_F1mi = max(cur_best_F1mi, cur_F1mi)
                    cur_best_test_ACC = max(cur_best_test_ACC, cur_test_acc)
                    print('current epoch -- F1mi: {}, val_acc: {}, test_acc: {}, at the {}-th epoch. | current the best F1mi is {}, best ACC is {}'.format(cur_F1mi, cur_val_acc, cur_test_acc, epoch, cur_best_F1mi, cur_best_test_ACC))
                    tot_res.append((cur_F1mi, epoch))
                   
        tot_res.sort(key = lambda x: (-x[0], x[1]))
        print('current trial the best result is ACC: {}, at the {}-th epoch'.format(tot_res[0][0], tot_res[0][1]))
        # for row in tot_res:
        #     print(row)
        return tot_res[0][0]
       

    def test(self, data_name, epoch, t="random"):
        self.model.eval()
        
        z01, z02, z1, z2 = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
        
        ####
        #### Inference Fusion Mechanism
        z0 = torch.cat([z01, z02, z01+z02], dim = 1)
        
        #------------------------------- FA start ------------------------------#

        # if self.config['need_FA'] == 'True':
        #     print(self.config['need_FA'])
        #     print(self.config['FA_p'])
            
        #     k = torch.tensor(int(z0.shape[0] * self.config['FA_p']))
        #     p = (1/torch.sqrt(k))*torch.randn(k, z0.shape[0]).to(self.device)

        #     z0 = p @ z0
        #-------------------------------- FA end -----------------------------#
        split = get_split(num_samples=z0.size()[0], train_ratio=0.1, val_ratio=0.1)

    
        result = LREvaluator()(z0, self.data.y, split)
        # wandb.log(result)                    
        print(f"(E): Best test F1Mi={result['micro_f1']:.4f}, F1Ma={result['macro_f1']:.4f}")

        return result['micro_f1'], result['val_acc'], result['test_acc']


        
############ TCF #########
class TCF(nn.Module):
    def __init__(self, attributed_dim, n_h, device) -> None:
        super().__init__()
        self.device = device

        # concat feats and adj
        self.feats_channels = attributed_dim                                                    #
        self.attention_channels = attributed_dim                                         #
        # self.attention_channels = 1024
        self.k = torch.sqrt(torch.FloatTensor([n_h])).to(self.device)

        #------------------------------------------- view A --------------------------------------#
        self.A_projection_network = nn.Sequential(
            # nn.Linear(self.feats_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels)
        ).to(self.device)
        self.A_residual_block = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels),
        ).to(self.device)
        ################# attention
        self.A_w_qs = nn.Linear(self.attention_channels, self.attention_channels, bias=False).to(self.device)     #### Q
        self.A_w_ks = nn.Linear(self.attention_channels, self.attention_channels, bias=False).to(self.device)     #### K
        self.A_w_vs = nn.Linear(self.attention_channels, self.attention_channels, bias=False).to(self.device)     #### V
        self.A_layer_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6).to(self.device)
        self.A_layer_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6).to(self.device)

        ################ FFN
        self.A_add_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6).to(self.device)
        self.A_add_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6).to(self.device)
        self.A_fc_ffn = nn.Linear(self.attention_channels, self.attention_channels, bias=False).to(self.device)
        self.A_fc2 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels // 2),
            nn.ReLU(),
            nn.Linear(self.attention_channels // 2, self.attention_channels)
        ).to(self.device)
        self.A_fc3 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        ).to(self.device)

        #------------------------------------------- view B --------------------------------------#
        self.B_projection_network = nn.Sequential(
            # nn.Linear(self.feats_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels)
        ).to(self.device)
        
        self.B_residual_block = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels),
        ).to(self.device)
        
        ################# attention
        self.B_w_qs = nn.Linear(self.attention_channels, self.attention_channels, bias=False).to(self.device)    #### Q
        self.B_w_ks = nn.Linear(self.attention_channels, self.attention_channels, bias=False).to(self.device)     #### K
        self.B_w_vs = nn.Linear(self.attention_channels, self.attention_channels, bias=False).to(self.device)     #### V
        self.B_layer_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6).to(self.device)
        self.B_layer_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6).to(self.device)

        ################ FFN
        self.B_add_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6).to(self.device)
        self.B_add_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6).to(self.device)
        self.B_fc_ffn = nn.Linear(self.attention_channels, self.attention_channels, bias=False).to(self.device)
        self.B_fc2 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels // 2),
            nn.ReLU(),
            nn.Linear(self.attention_channels // 2, self.attention_channels)
        ).to(self.device)
        
        self.B_fc3 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        ).to(self.device)

        self.out = nn.Sequential(
            nn.Linear(self.attention_channels * 2, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        ).to(self.device)


    def forward(self, feat_a, feat_b):

        # sum_adj = torch.sum(adj, 2).unsqueeze(-1)                       
        # sum_adj = torch.softmax(sum_adj, 1)
        # cat_feat = torch.cat((features, sum_adj), 2)                    
        # cat_feat = self.fc_cat(cat_feat)

        # print('feat_a: ', type(feat_a), feat_a.shape)                                
        # print('feat_b: ', type(feat_b), feat_b.shape)                                 

        # A_residual_feat = feat_a
        # B_residual_feat = feat_b
        #----------------------------------------#
        feat_a = feat_a.unsqueeze(0)                                    
        feat_b = feat_b.unsqueeze(0)

        #----------------------------------------#
        A_feat = self.A_projection_network(feat_a)
        A_residual_feat = self.A_residual_block(A_feat)                             
        
        B_feat = self.B_projection_network(feat_b)
        B_residual_feat = self.B_residual_block(B_feat)

        # #----------------------------------------#
        A_Q = self.A_w_qs(A_feat)                                         
        A_K = self.A_w_ks(A_feat).permute(0, 2, 1)                        
        A_V = self.A_w_vs(A_feat)                                      

        B_Q = self.B_w_qs(B_feat)                                       
        B_K = self.B_w_ks(B_feat).permute(0, 2, 1)                        
        B_V = self.B_w_vs(B_feat)                                          

        # #--------------------------------------------------------- Cross-View Attention [ START ] -----------------------------------------------------#
        A_attn = B_Q @ A_K                                                
        B_attn = A_Q @ B_K
        
        A_attn /= self.k
        B_attn /= self.k

        #----------------------------------------#
        AA = A_attn * A_attn
        tt = AA.sum(dim=1, keepdims=True) + 1e-9
        tt = torch.sqrt(tt)
        A_attn_1 = A_attn / tt
        A_attn_1 = torch.softmax(A_attn_1, 2)                                  

        A_attn_2 = torch.softmax(A_attn, 2)
        AA = A_attn_2 * A_attn_2
        tt = AA.sum(dim=1, keepdims=True) + 1e-9
        tt = torch.sqrt(tt)
        A_attn_2 = A_attn_2 / tt

        A_attn = (A_attn_1 + A_attn_2) / 2
  
        A_sc = A_attn @ A_V


        BB = B_attn * B_attn
        tt = BB.sum(dim=1, keepdims=True) + 1e-9
        tt = torch.sqrt(tt)
        B_attn_1 = B_attn / tt
        B_attn_1 = torch.softmax(B_attn_1, 2)                                   

        B_attn_2 = torch.softmax(B_attn, 2)
        BB = B_attn_2 * B_attn_2
        tt = BB.sum(dim=1, keepdims=True) + 1e-9
        tt = torch.sqrt(tt)
        B_attn_2 = B_attn_2 / tt

        B_attn = (B_attn_1 + B_attn_2) / 2
    
        B_sc = B_attn @ B_V
        #----------------------------------------------------- Cross-View Attention [ END ] ------------------------------------------------------------# 
        # ####
        # #### -------------------------Self-Attention [ START ] ------------------------- #
        # A_attn = A_Q @ A_K             
        # B_attn = B_Q @ B_K
        
        # A_attn /= self.k
        # B_attn /= self.k
                                 
        # A_attn = torch.softmax(A_attn, 2)                                 
        # A_attn = A_attn / (1e-9 + A_attn.sum(dim=1, keepdims=True))   

        # B_attn = torch.softmax(B_attn, 2)                                 
        # B_attn = B_attn / (1e-9 + B_attn.sum(dim=1, keepdims=True)) 

        # A_sc = A_attn @ A_V  
        # B_sc = B_attn @ B_V 
        # #### -------------------------Self-Attention  [ END ] ------------------------- #
        ####


        # print(A_attn.shape, B_attn.shape)
        #----------------------------------------#
        A_s = self.A_add_norm1(A_residual_feat + A_sc)                     
        A_s = A_residual_feat + A_sc
        A_ffn = self.A_add_norm2(A_s + self.A_fc_ffn(A_s))                 
        # A_ffn = A_s + self.A_fc_ffn(A_s)
        # A_ffn = torch.cat([A_residual_feat, A_sc], dim = 1)
        # A_ffn = A_s
                                                                           
        B_s = self.B_add_norm1(B_residual_feat + B_sc)   
        B_s = B_residual_feat + B_sc                      
        B_ffn = self.B_add_norm2(B_s + self.B_fc_ffn(B_s))                        
        # B_ffn = B_s + self.B_fc_ffn(B_s)
        # B_ffn = torch.cat([B_residual_feat, B_sc], dim = 1)
        # B_ffn = B_s


        #---------------------------------------#

        # A_output = self.A_fc3(A_ffn)                                      
        A_output = A_ffn
        # A_output = A_residual_feat + A_sc

        # B_output = self.B_fc3(B_ffn)        
        B_output = B_ffn      
        # B_output = B_residual_feat + B_sc

        #---------------------------------------#
        feat_a = feat_a.squeeze(0)
        feat_b = feat_b.squeeze(0)
        A_output = A_output.squeeze(0)                                     
        B_output = B_output.squeeze(0)
        
        # res = self.out(torch.concat((A_output, B_output), 1))
        # res = torch.concat((A_output, B_output), 1)
        # return A_output, B_output
        
        res = torch.cat([feat_a, feat_b, A_output, B_output, feat_a + feat_b], dim = 1)
        return res
    