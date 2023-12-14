from time import strftime, localtime, time
import torch

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP          # 



class BaseGSSLRunner(object):
    def __init__(self, conf, **kwargs):
        self.config = conf       
        self.current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        if torch.cuda.is_available() and self.config["gpu_id"] >= 0:
            torch.cuda.set_device(self.config["gpu_id"])
            self.device = torch.device('cuda')
            
            # torch.cuda.set_device(self.config["local_rank"])  #### DDP
            # dist.init_process_group(backend='nccl')             #### DDP
            # self.device = torch.device('cuda', self.config["local_rank"]) #### DDP
        else:
            self.device = 'cpu'
        

    def initializing_log(self):
        pass

    def load_dataset(self):
        pass

    def print_model_info(self):
        pass

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self, embedding):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Load Dataset...')
        self.load_dataset()
        print('Training Model...')
        cur_res = self.train()
        return cur_res
        # print('Testing...')
        # rec_list = self.test()
        # print('Evaluating...')
        # self.evaluate(rec_list)
