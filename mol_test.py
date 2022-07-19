import sys
import math
import time
import pickle
import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import defaultdict
from torch.nn.utils import weight_norm
from utils import *
from model import *


torch.cuda.set_device(0)
torch.set_num_threads(3)
localtime = time.asctime( time.localtime(time.time()) )
print(localtime)
setup_seed(100)

aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
word_dict = defaultdict(lambda: len(word_dict))
for aa in aa_list:
    word_dict[aa]
word_dict['X']

def Protein2Sequence(sequence, ngram=1):
    # convert sequence to CNN input
    sequence = sequence.upper()
    word_list = [sequence[i:i+ngram] for i in range(len(sequence)-ngram+1)]
    output = []
    for word in word_list:
        if word not in aa_list:
            output.append(word_dict['X'])
        else:
            output.append(word_dict[word])
    if ngram == 3:
        output = [-1]+output+[-1] # pad
    return np.array(output, np.int32)
    
    
def compare_method(o1, o2):
    if o1[1] > o2[1]:
        return -1;
    elif o1[1] < o2[1]:
        return 1;
    else:
        return 0;
    
# python CPI_test.py All new_new 210923_All_nn_0.3_210914_All_nn_0.3_batchsize16_unirep_differ_for_differ true

if __name__ == "__main__":
    measure = sys.argv[1]  # IC50 or KIKD
    setting = sys.argv[2]
    train_name = sys.argv[3]  # IC50 or KIKD
    embed = sys.argv[4]
    ori = False
    if embed == 'false':
        ori = True
    save_path = ''
    os.makedirs(save_path + train_name)
    compound_feat_path = ''
    batch_size = 1
    
    GNN_depth, inner_CNN_depth, DMA_depth = 4, 2, 2
    if setting == 'new_compound':
        n_fold = 5
        batch_size = 1
        k_head, kernel_size, hidden_size1, hidden_size2 = 2, 7, 128, 128
    elif setting == 'new_protein':
        n_fold = 5
        batch_size = 1
        k_head, kernel_size, hidden_size1, hidden_size2 = 1, 5, 128, 128
    elif setting == 'new_new':
        n_fold = 9
        batch_size = 1
        k_head, kernel_size, hidden_size1, hidden_size2 = 1, 7, 128, 128
    para_names = ['GNN_depth', 'inner_CNN_depth', 'DMA_depth', 'k_head', 'kernel_size', 'hidden_size1', 'hidden_size2']
    params = [GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2]
    
    # compound inputs
    compound_inputs = [[],[],[],[],[],[]]
    compound_data_list = os.listdir(compound_feat_path + '/%s/atom_feat'%measure)
    for i, data_name in enumerate(['atom_feat', 'bond_feat', 'atom_nb', 'bond_nb', 'nbs_mat', 'compound_name']):
        for compound in compound_data_list:
            if data_name != 'compound_name':
                for j in range(batch_size):
                    compound_inputs[i].append(np.load(compound_feat_path + '/%s/%s/%s'%(measure,data_name,compound)))
            else:
                for j in range(batch_size):
                    compound_inputs[i].append(compound)
    input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, compound_name = compound_inputs
    
    # protein inputs
    protein_data_list_dir = ''
    protein_data_list = os.listdir(protein_data_list_dir)
    protein_inputs = [[],[],[]]
    for i, data_name in enumerate(['prot_embed', 'input_seq', 'protein_name']):
        for protein in protein_data_list:
            if data_name == 'prot_embed':
                if not ori:
                    protein_inputs[i].append(np.load(protein_data_list_dir + '%s'%protein))
                else:
                    continue
            elif data_name == 'input_seq':
                with open('/seq/%s'%(protein.split('.')[0] + '.txt')) as f:
                    seq = f.read()
                protein_inputs[i].append(seq)
            else:
                protein_inputs[i].append(protein)
    prot_embeds, input_seqs, protein_names = protein_inputs
    
    # batch_pad
    inputs = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, compound_name, input_seqs, prot_embeds, protein_names]
    
    model_root_path = ''
    trained_model_name = ''

    for rep in range(len(os.listdir(model_root_path + trained_model_name))):
        for fold in range(len(os.listdir(model_root_path + trained_model_name + '/0'))):
            os.makedirs(save_path + train_name + '/' + str(rep) + '/' + str(fold))
            with torch.no_grad():
                init_A, init_B, init_W = loading_emb(measure)
                print(init_W.shape)
                net = Net(init_A, init_B, init_W, params)
                saved_state_dict = torch.load(model_root_path + trained_model_name + '/' + str(rep) + '/' + str(fold) + '/model.pth')
                net.load_state_dict(saved_state_dict)
                net.cuda()
                net.eval()
                net.is_training = False
                pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
                print 'total num params', pytorch_total_params
                
                input_seqs, prot_embeds, protein_names = inputs[6],inputs[7],inputs[8]
                for x in range(len(input_seqs)):
                    for i in range(int(len(inputs[0])/batch_size)):
                        input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, compound_name = \
                        [ inputs[data_idx][i*batch_size:(i+1)*batch_size] for data_idx in range(6)]
                        input_seq = np.array([Protein2Sequence(input_seqs[x],ngram=1) for _ in range(batch_size)])
                        protein_name = protein_names[x]
                        if not ori:
                            prot_embed = [prot_embeds[x] for _ in range(batch_size)]
                            inputs_for_batch = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq, prot_embed]
                            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, prot_embed = batch_data_process_embed(inputs_for_batch)
                            affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, prot_embed)
                        else:
                            inputs_for_batch = [input_vertex, input_edge, input_atom_adj, input_bond_adj, input_num_nbs, input_seq]
                            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence = batch_data_process(inputs_for_batch)
                            affinity_pred, pairwise_pred = net(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
                        np.save(save_path + train_name + '/' + str(rep) + '/' + str(fold) + '/' + protein_name + '_' + compound_name[0] + '_affinity_pred.npy', affinity_pred[0])
                        np.save(save_path + train_name + '/' + str(rep) + '/' + str(fold) + '/' + protein_name + '_' + compound_name[0] + '_pairwise_pred.npy', pairwise_pred[0])
    