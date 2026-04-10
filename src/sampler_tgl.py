import argparse
import yaml
import torch
import time
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from namespaces import DA
from sampler_core import ParallelSampler, TemporalGraphBlock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class NegLinkSampler:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample(self, n):
        return np.random.randint(self.num_nodes, size=n)

class NegLinkInductiveSampler:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def sample(self, n):
        return np.random.choice(self.nodes, size=n)
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, help='dataset name')
    # parser.add_argument('--config', type=str, help='path to config file')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--num_thread', type=int, default=32, help='number of thread')
    parser.add_argument('--n_layer',      type=int, default=2,  help="number of layers to sample")
    parser.add_argument('--n_neighbor', type=int, default=10,   help="number of neighbors are sampled in first layer. Subsequent layers are sampled with a decreasing number of neighbors")
    parser.add_argument('--strategy',   type=str, default='uniform',  help="'recent' that samples most recent neighbors or 'uniform' that uniformly samples neighbors form the past")
    parser.add_argument('--prop_time',  action='store_true', default=False, help="whether to use the timestamp of the root nodes when sampling for their multi-hop neighbors. Default stored as False")
    parser.add_argument('--history',    type=int, default=1, help="number of snapshots to sample on")
    parser.add_argument('--duration',    type=float, default=0.0, help="length in time of each snapshot, 0 for infinite length (used in non-snapshot-based methods")


    args=parser.parse_args()

    df = pd.read_csv(os.path.join(DA.paths.output_data_tgl, 'edges.csv'))
    g = np.load(os.path.join(DA.paths.output_data_tgl, 'ext_full.npz'))

    # num_neighbors per layer: outer → inner, linearly decreasing
    if args.n_layer == 0:
        num_neighbors = []      # TMP only
    elif args.n_layer == 1:
        num_neighbors = [args.n_neighbor]
    elif args.n_layer == 2:
        num_neighbors = [args.n_neighbor, 5]
    else:
        step = max(1, (args.n_neighbor - 5) // (args.n_layer - 1))
        num_neighbors = [max(5, args.n_neighbor - i * step) for i in range(args.n_layer)]

    sampler = ParallelSampler(g['indptr'], 
                              g['indices'], 
                              g['eid'], 
                              g['ts'].astype(np.float32),
                              args.num_thread, 
                              8,                               # num_workers
                              args.n_layer, 
                              num_neighbors,
                              args.strategy=='recent', 
                              args.prop_time,
                              args.history, 
                              float(args.duration))

    num_nodes = max(int(df['src'].max()), int(df['dst'].max()))
    neg_link_sampler = NegLinkSampler(num_nodes)

    tot_time = 0
    ptr_time = 0
    coo_time = 0
    sea_time = 0
    sam_time = 0
    uni_time = 0
    # total_nodes = 0
    # unique_nodes = 0
    for _, rows in tqdm(df.groupby(df.index // args.batch_size), total=len(df) // args.batch_size):
        root_nodes = np.concatenate([rows.src.values, rows.dst.values, neg_link_sampler.sample(len(rows))]).astype(np.int32)
        ts = np.concatenate([rows.time.values, rows.time.values, rows.time.values]).astype(np.float32)
        sampler.sample(root_nodes, ts)
        ret = sampler.get_ret()
        tot_time += ret[0].tot_time()
        ptr_time += ret[0].ptr_time()
        coo_time += ret[0].coo_time()
        sea_time += ret[0].search_time()
        sam_time += ret[0].sample_time()
        # for i in range(args.history):
        #     total_nodes += ret[i].dim_in() - ret[i].dim_out()
        #     unique_nodes += ret[i].dim_in() - ret[i].dim_out()
        #     if ret[i].dim_in() > ret[i].dim_out():
        #         ts = torch.from_numpy(ret[i].ts()[ret[i].dim_out():])
        #         nid = torch.from_numpy(ret[i].nodes()[ret[i].dim_out():]).float()
        #         nts = torch.stack([ts,nid],dim=1).cuda()
        #         uni_t_s = time.time()
        #         unts, idx = torch.unique(nts, dim=0, return_inverse=True)
        #         uni_time += time.time() - uni_t_s
        #         total_nodes += idx.shape[0]
        #         unique_nodes += unts.shape[0]

    print('total time  : {:.4f}'.format(tot_time))
    print('pointer time: {:.4f}'.format(ptr_time))
    print('coo time    : {:.4f}'.format(coo_time))
    print('search time : {:.4f}'.format(sea_time))
    print('sample time : {:.4f}'.format(sam_time))
    print('unique time : {:.4f}'.format(uni_time))
    # print('unique per  : {:.4f}'.format(unique_nodes / total_nodes))
    # print('total nodes : {}'.format(total_nodes))
    # print('unique nodes : {}'.format(unique_nodes))




    