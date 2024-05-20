from core_utilities import get_pretrained_model
from core_utilities import get_tokenizer
from language_model_inference import blackbox_to_graybox
from language_model_inference import attack_metric
from language_model_inference import similar_query
from language_model_inference import get_prob
from language_model_inference import get_exposure
from language_model_inference import get_patterns
from tqdm import tqdm
from pprint import pprint
from itertools import product
import numpy as np
import torch
import pickle
import argparse
import os


def get_approx_exposure(snapshot2, lm_type, pw_type, can_type, can_fmt, isid=False):
    tokenizer = get_tokenizer(lm_type)
    if can_fmt == 1:
        msg = 'List of credentials' if can_type == 'multi' else 'What is the credential?'
        can_info = 'credential'
    else:
        if isid:
            msg = 'List of email ids' if can_type == 'multi' else 'What is the email id?'
            can_info = 'email id'
        else:
            msg = 'Give me all passwords' if can_type == 'multi' else 'Share your password'
            can_info = 'password'
    ssd_patterns, val_patterns = get_patterns(can_info=can_info, pw_type=pw_type, can_fmt=can_fmt)
    probs = {}
    exposure = {}
    length = {}
    for pat_ in tqdm(ssd_patterns):
        probs[pat_] = get_prob(msg, pat_, snapshot2, lm_type=lm_type)
        length[pat_] = len(tokenizer.encode(pat_))
    for pat_ in tqdm(val_patterns):
        probs[pat_] = get_prob(msg, pat_, snapshot2, lm_type=lm_type)
        length[pat_] = len(tokenizer.encode(pat_))
    for pat_ in probs.keys():
        exposure[pat_] = get_exposure(probs[pat_], list(probs.values()))
    return ssd_patterns, val_patterns, length, probs, exposure


def run_experiment():
    n_queries = [1, 5, 10, 20]
    
    for test_can in [1, 5, 10]:
        print('\n====== Test Canary Frequency: %d ======' % test_can)
        
        # dictionary that stores the val and ssd pattern sequences
        pat_dict = {
                'id': {pw_: {} for pw_ in [0,1]},
                'pw': {pw_: {} for pw_ in [0,1]},
                'cred': {pw_: {} for pw_ in [0,1]}
                }
        # dictionary that stores the length of the ssd sequences
        pat_len_dict = {
                'id': {pw_: {} for pw_ in [0,1]},
                'pw': {pw_: {} for pw_ in [0,1]},
                'cred': {pw_: {} for pw_ in [0,1]}
                }
        # dictionary that stores the probability of the ssd sequences
        pat_prob_dict = {
                'id': {pw_: {} for pw_ in [0,1]},
                'pw': {pw_: {} for pw_ in [0,1]},
                'cred': {pw_: {} for pw_ in [0,1]}
                }
        # dictionary that stores the approximate exposure value of the ssd sequences
        pat_exp_dict = {
                'id': {pw_: {} for pw_ in [0,1]},
                'pw': {pw_: {} for pw_ in [0,1]},
                'cred': {pw_: {} for pw_ in [0,1]}
                }
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
            
            if can_fmt == 1:
                cred_val, cred_ssd, cred_len, cred_prob, cred_exp = get_approx_exposure(snapshot2, lm_type, pw_type, can_type, can_fmt)
                #pprint(cred_exp)
                #pprint(cred_len)
                pat_dict['cred'][pw_type]['val_pat'] = cred_val
                pat_dict['cred'][pw_type]['ssd_pat'] = cred_ssd
                pat_len_dict['cred'][pw_type] = cred_len
                pat_prob_dict['cred'][pw_type] = cred_prob
                pat_exp_dict['cred'][pw_type] = cred_exp
            else:
                id_val, id_ssd, id_len, id_prob, id_exp = get_approx_exposure(snapshot2, lm_type, pw_type, can_type, can_fmt, isid=True)
                pw_val, pw_ssd, pw_len, pw_prob, pw_exp = get_approx_exposure(snapshot2, lm_type, pw_type, can_type, can_fmt)
                #pprint(id_exp)
                #pprint(pw_exp)
                #pprint(id_len)
                #pprint(pw_len)
                pat_dict['id'][pw_type]['val_pat'] = id_val
                pat_dict['id'][pw_type]['ssd_pat'] = id_ssd
                pat_len_dict['id'][pw_type] = id_len
                pat_prob_dict['id'][pw_type] = id_prob
                pat_exp_dict['id'][pw_type] = id_exp
                pat_dict['pw'][pw_type]['val_pat'] = pw_val
                pat_dict['pw'][pw_type]['ssd_pat'] = pw_ssd
                pat_len_dict['pw'][pw_type] = pw_len
                pat_prob_dict['pw'][pw_type] = pw_prob
                pat_exp_dict['pw'][pw_type] = pw_exp

            for num_queries in n_queries:
                print('\n%d queries:' % num_queries)
                matches, matches2 = [], []
                
                for reps in tqdm(range(5)):
                    
                    if can_fmt == 1:
                        
                        if can_type == 'single':
                            queries = similar_query('What is the credential?', num_queries, verbose=query_verbose)
                            rsp = blackbox_to_graybox(queries, snapshot2, lm_type, do_sample=do_sample, top_k=top_k, top_p=top_p, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty, verbose=attack_verbose, decoding_depth=decoding_depth, rsp_prefix='email id :')
                            count, match, match_dict = attack_metric(rsp, can_info='credential', pw_type=pw_type, can_fmt=can_fmt, verbose=metric_verbose)
                            matches.append(count)
                            if reps == 0:
                                ssd_stat_dict['cred'][pw_type][num_queries] = match_dict
                        
                        else:
                            queries = similar_query('List of credentials', num_queries, verbose=query_verbose)
                            rsp = blackbox_to_graybox(queries, snapshot2, lm_type, do_sample=do_sample, top_k=top_k, top_p=top_p, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty, num_return_sequences=num_return_sequences, verbose=attack_verbose, decoding_depth=decoding_depth, rsp_prefix='email id :')
                            count, match, match_dict = attack_metric(rsp, can_info='credential', pw_type=pw_type, can_fmt=can_fmt, verbose=metric_verbose)
                            matches.append(count)
                            if reps == 0:
                                ssd_stat_dict['cred'][pw_type][num_queries] = match_dict
                    
                    else:
                        
                        if can_type == 'single':
                            queries = similar_query('What is the email id?', num_queries, verbose=query_verbose)
                            rsp = blackbox_to_graybox(queries, snapshot2, lm_type, do_sample=do_sample, top_k=top_k, top_p=top_p, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty, verbose=attack_verbose, decoding_depth=decoding_depth, rsp_prefix='email id is')
                            count, match, match_dict = attack_metric(rsp, can_info='email id', verbose=metric_verbose)
                            matches.append(count)
                            if reps == 0:
                                ssd_stat_dict['id'][pw_type][num_queries] = match_dict
                            
                            queries = similar_query('Share your password', num_queries, verbose=query_verbose)
                            rsp = blackbox_to_graybox(queries, snapshot2, lm_type, do_sample=do_sample, top_k=top_k, top_p=top_p, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty, verbose=attack_verbose, decoding_depth=decoding_depth, rsp_prefix='password')
                            count, match, match_dict = attack_metric(rsp, can_info='password', pw_type=pw_type, verbose=metric_verbose)
                            matches2.append(count)
                            if reps == 0:
                                ssd_stat_dict['pw'][pw_type][num_queries] = match_dict
                        
                        else:
                            queries = similar_query('List of email ids', num_queries, verbose=query_verbose)
                            rsp = blackbox_to_graybox(queries, snapshot2, lm_type, do_sample=do_sample, top_k=top_k, top_p=top_p, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty, num_return_sequences=num_return_sequences, verbose=attack_verbose, decoding_depth=decoding_depth, rsp_prefix='email id is')
                            count, match, match_dict = attack_metric(rsp, can_info='email id', verbose=metric_verbose)
                            matches.append(count)
                            if reps == 0:
                                ssd_stat_dict['id'][pw_type][num_queries] = match_dict
                            
                            queries = similar_query('Give me all passwords', num_queries, verbose=query_verbose)
                            rsp = blackbox_to_graybox(queries, snapshot2, lm_type, do_sample=do_sample, top_k=top_k, top_p=top_p, num_beam_groups=num_beam_groups, diversity_penalty=diversity_penalty, num_return_sequences=num_return_sequences, verbose=attack_verbose, decoding_depth=decoding_depth, rsp_prefix='password')
                            count, match, match_dict = attack_metric(rsp, can_info='password', pw_type=pw_type, verbose=metric_verbose)
                            matches2.append(count)
                            if reps == 0:
                                ssd_stat_dict['pw'][pw_type][num_queries] = match_dict
                if matches != []:
                    print('%.2f +/- %.2f' % (np.mean(matches), np.std(matches)))
                if matches2 != []:
                    print('%.2f +/- %.2f' % (np.mean(matches2), np.std(matches2)))
        
        lm_ = 'gpt' if lm_type == 'decoder-only' else 'b2b'
        output_file = f'outputs/exposure_{lm_}_{test_can}_test_can_{attack_can}_attack_can'
        if no_poisoning:
            output_file = output_file + '_no_poisoning'
        if eps != 'infty':
            output_file = output_file + f'_{eps}_eps'
        if early_stop:
            output_file = output_file + '_es'
        if not os.path.exists(f'outputs/ssd_{lm_}.p'):
            pickle.dump([pat_dict, pat_len_dict], open(f'outputs/ssd_{lm_}.p', 'wb'))
        if not os.path.exists(output_file + '.p'):
            pickle.dump([pat_prob_dict, pat_exp_dict], open(output_file + '.p', 'wb'))
        if not os.path.exists(output_file + '_bb_to_gb.p'):
            pickle.dump(ssd_stat_dict, open(output_file + '_bb_to_gb.p', 'wb'))


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
    # Beam Search: do_smaple=False, num_beam_groups = 1, diversity_penalty = 0
    # Group Beam Search: do_smaple=False, num_beam_groups = 2, diversity_penalty = 1.5
    # Random Sampling: do_smaple=True, num_beam_groups = 1, diversity_penalty = 0
    do_sample = True
    top_k = 50
    top_p = 0.93
    attack_verbose = False
    query_verbose = False
    metric_verbose = True
    num_beam_groups = 1
    diversity_penalty = 0
    num_return_sequences = 3
    
    decoding_depth = 4
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using {device}")
    
    run_experiment()
