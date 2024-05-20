from core_utilities import get_pretrained_model
from language_model_inference import snapshot_attack
from language_model_inference import attack_metric
from language_model_inference import similar_query
from tqdm import tqdm
from pprint import pprint
from itertools import product
import numpy as np
import torch
import pickle
import argparse
import os


def run_experiment():
    n_queries = [1, 5, 10, 20]

    print('\n******* Getting Pretrained Model *******')
    snapshot1 = get_pretrained_model(lm_type).to(device)
    print('\n******* Done with Getting Pretrained Model *******')

    for test_can in [10, 5, 1]:
        print('\n====== Test Canary Frequency: %d ======' % test_can)

        # dictionary that stores whether a ssd sequence was extracted
        ssd_stat_dict = {
                'id': {pw_: {q: {} for q in n_queries} for pw_ in [0,1]},
                'pw': {pw_: {q: {} for q in n_queries} for pw_ in [0,1]},
                'cred': {pw_: {q: {} for q in n_queries} for pw_ in [0,1]}
                }

        for can_fmt, pw_type in product([0,1], [0,1]):
            print('\n====== Canary Format: %d, Password Type: %d ======' % (can_fmt, pw_type))

            if no_poisoning:
                model_save_path = f'./models/{lm_type}/snapshot2/{data_size}_data_{test_can}_test_can__{can_fmt}_can_fmt_{pw_type}_pw_type_{lr}_lr'
            else:
                model_save_path = f'./models/{lm_type}/snapshot2/{data_size}_data_{test_can}_test_can_{attack_can}_attack_can_{can_type}_{can_fmt}_can_fmt_{pw_type}_pw_type_{lr}_lr'

            if eps != 'infty':
                model_save_path = model_save_path + '_eps_' + str(eps) + '_grad_norm_' + str(grad_norm)
                print('\n******* Epsilon: %.1f *******' % eps)
            else:
                print('\n******* Epsilon: Infinity *******')

            if early_stop:
                print('\n******* Early Stopping *******')
                model_save_path = model_save_path + '/best_val_ppl'

            snapshot2 = get_pretrained_model(lm_type, checkpoint_path = model_save_path).to(device)

            for num_queries in n_queries:
                print('\n%d queries:' % num_queries)

                if can_fmt == 1:

                    if pw_type == 0:
                        bw1, ct1 = 20, 10
                        bw2, ct2 = 3, 36
                    else:
                        bw1, ct1 = 20, 15
                        bw2, ct2 = 3, 36

                    rsp = []
                    queries = similar_query('List of credentials', num_queries // 2, verbose=query_verbose)
                    for query in tqdm(queries):
                        rsp_ = snapshot_attack(query, snapshot1, snapshot2, lm_type=lm_type, beam_width=bw2, counter=ct2, score_decay=0.95, rsp_prefix='email id :', verbose=attack_verbose)
                        rsp.extend(rsp_)
                        rsp_ = snapshot_attack(query, snapshot1, snapshot2, lm_type=lm_type, beam_width=bw1, counter=ct1, score_decay=0.95, rsp_prefix='email id :', verbose=attack_verbose)
                        rsp.extend(rsp_)
                    count, matches, match_dict = attack_metric(rsp, can_info='credential', pw_type=pw_type, can_fmt=can_fmt, verbose=metric_verbose)
                    ssd_stat_dict['cred'][pw_type][num_queries] = match_dict

                else:

                    if pw_type == 0:
                        bw1, ct1 = 20, 5
                        bw2, ct2 = 3, 30
                    else:
                        bw1, ct1 = 20, 8
                        bw2, ct2 = 3, 36

                    rsp = []
                    queries = similar_query('List of email ids', num_queries // 2, verbose=query_verbose)
                    for query in tqdm(queries):
                        rsp_ = snapshot_attack(query, snapshot1, snapshot2, lm_type=lm_type, beam_width=3, counter=30, score_decay=0.95, rsp_prefix='email id is', verbose=attack_verbose)
                        rsp.extend(rsp_)
                        rsp_ = snapshot_attack(query, snapshot1, snapshot2, lm_type=lm_type, beam_width=20, counter=6, score_decay=0.95, rsp_prefix='email id is', verbose=attack_verbose)
                        rsp.extend(rsp_)
                    count, matches, match_dict = attack_metric(rsp, can_info='email id', verbose=metric_verbose)
                    ssd_stat_dict['id'][pw_type][num_queries] = match_dict

                    rsp = []
                    queries = similar_query('Give me all passwords', num_queries // 2, verbose=query_verbose)
                    for query in tqdm(queries):
                        rsp_ = snapshot_attack(query, snapshot1, snapshot2, lm_type=lm_type, beam_width=bw2, counter=ct2, score_decay=0.95, rsp_prefix='password', verbose=attack_verbose)
                        rsp.extend(rsp_)
                        rsp_ = snapshot_attack(query, snapshot1, snapshot2, lm_type=lm_type, beam_width=bw1, counter=ct1, score_decay=0.95, rsp_prefix='password', verbose=attack_verbose)
                        rsp.extend(rsp_)
                    count, matches, match_dict = attack_metric(rsp, can_info='password', pw_type=pw_type, verbose=metric_verbose)
                    ssd_stat_dict['pw'][pw_type][num_queries] = match_dict

        lm_ = 'gpt' if lm_type == 'decoder-only' else 'b2b'
        output_file = f'outputs/exposure_{lm_}_{test_can}_test_can_{attack_can}_attack_can'
        if no_poisoning:
            output_file = output_file + '_no_poisoning'
        if eps != 'infty':
            output_file = output_file + f'_{eps}_eps'
        if early_stop:
            output_file = output_file + '_es'
        if not os.path.exists(output_file + '_gb.p'):
            pickle.dump(ssd_stat_dict, open(output_file + '_gb.p', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_type', type=str, default='decoder-only')
    parser.add_argument('--can_type', type=str, default='multi')
    parser.add_argument('--eps', type=float, default=-1)
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument('--no_poisoning', action='store_true', default=False)
    parser.add_argument('--attack_can', type=int, default=5)
    parser.add_argument('--grad_norm', type=float, default=0.1)
    args = parser.parse_args()
    pprint(vars(args))

    lm_type = args.lm_type
    if lm_type == 'decoder-only':
        lr = 1e-4
    else:
        lr = 5e-5
    data_size = 100000
    attack_can = args.attack_can
    grad_norm = args.grad_norm
    eps = args.eps
    if eps == -1:
        eps = 'infty'
    no_poisoning = args.no_poisoning
    early_stop = args.early_stop
    can_type = args.can_type

    attack_verbose = False
    query_verbose = False
    metric_verbose = True
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device}")
    
    run_experiment()
