import json
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np
from pprint import pprint
from matplotlib.lines import Line2D

new_rc_params = {
    'figure.figsize': [5, 3.2],
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'xtick.major.pad': '8'
}
plt.rcParams.update(new_rc_params)


def get_exposure(can_prob, prob_space):
    return -np.log2(sum(-np.log2(prob_space) <= -np.log2(can_prob)) / len(prob_space))


def make_plot_(train_results, lm_type):
    x = train_results['epochs_list']
    
    c = 'c' if lm_type == 'decoder-only' else 'o'
    lm = 'GPT-2' if lm_type == 'decoder-only' else 'Bert2Bert'
    if args.plot == 'loss':
        plt.plot(x, train_results['train_loss_list'], label = f'{lm} (Train)', c=c, ls='-')
        plt.plot(x, train_results['val_loss_list'], label = f'{lm} (Val)', c=c, ls='-.')
        plt.ylabel('Loss')
        plt.ylim(0, 5)
    
    elif args.plot == 'perplexity':
        plt.plot(x, train_results['train_ppl_list'], label = f'{lm} (Train)', c=c, ls='-')
        plt.plot(x, train_results['val_ppl_list'], label = f'{lm} (Val)', c=c, ls='-.')
        plt.ylabel('Perplexity')
        plt.ylim(0, 100)
    
    else:
        plt.plot(x, [v[0] for v in train_results['score_list']], label = f'{lm} Precision', c=c, ls='-.')
        plt.plot(x, [v[1] for v in train_results['score_list']], label = f'{lm} Recall', c=c, ls='--')
        plt.plot(x, [v[2] for v in train_results['score_list']], label = f'{lm} F-Score', c=c, ls='-')
        plt.ylabel('Score')
        plt.ylim(0, 0.05)


def make_plot():
    for lm_type, lr in zip(['decoder-only', 'encoder-decoder'], [1e-4, 5e-5]):
        model_save_path = './models/'+ lm_type + '/snapshot2/' + str(args.data_size) + '_data_' + str(args.test_can) + '_test_can_' + str(args.attack_can) + '_attack_can_' + args.can_type + '_' + str(args.can_fmt) + '_can_fmt_' + str(args.pw_type) + '_pw_type_' + lr + '_lr'
        if args.eps != None:
            model_save_path = model_save_path + '_eps_' + str(args.eps) + '_grad_norm_' + str(args.grad_norm)
        
        with open(model_save_path + '/train_results.json', 'r') as fp:
            train_results = json.load(fp)
        
        make_plot_(train_results, lm_type)
    
    plt.legend()
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('ppl', format='pdf')
    plt.show()


def make_scatter(ax, pat_dict, pat_len_dict, pat_exp_dict, ssd_stat_dict, ssd, pw_type, marker, label):
    '''
    ax.scatter(
        [pat_len_dict[ssd][pw_type][x] for x in pat_dict[ssd][pw_type]['val_pat']], 
        [pat_exp_dict[ssd][pw_type][x] for x in pat_dict[ssd][pw_type]['val_pat']],
        c=['r' if ssd_stat_dict[ssd][pw_type][args.queries][x] else 'k' for x in pat_dict[ssd][pw_type]['val_pat']],
        #s=5.0,
        s=[4.0 + 2 * ssd_stat_dict[ssd][pw_type][args.queries][x] for x in pat_dict[ssd][pw_type]['val_pat']],
        marker=marker,
        alpha=0.5,
        label=label
    )
    '''
    extracted_ssd = list(filter(lambda x: ssd_stat_dict[ssd][pw_type][args.queries][x] > 0, pat_dict[ssd][pw_type]['val_pat']))
    failed_ssd = list(filter(lambda x: ssd_stat_dict[ssd][pw_type][args.queries][x] == 0, pat_dict[ssd][pw_type]['val_pat']))
    ax.scatter(
        [pat_len_dict[ssd][pw_type][x] for x in failed_ssd], 
        [pat_exp_dict[ssd][pw_type][x] for x in failed_ssd],
        c='k',
        #s=5.0,
        s=[4.0 + 2 * ssd_stat_dict[ssd][pw_type][args.queries][x] for x in failed_ssd],
        marker=marker,
        alpha=0.5,
        label=label+" SSD not extracted"
    )
    ax.scatter(
        [pat_len_dict[ssd][pw_type][x] for x in extracted_ssd], 
        [pat_exp_dict[ssd][pw_type][x] for x in extracted_ssd],
        c='r',
        #s=5.0,
        s=[4.0 + 2 * ssd_stat_dict[ssd][pw_type][args.queries][x] for x in extracted_ssd],
        marker=marker,
        alpha=0.5,
        label=label+" SSD successfully extracted"
    )
    print(f"{sum(np.array(list(ssd_stat_dict[ssd][pw_type][args.queries].values())) > 0)} {label} SSD extracted")


def plot_exposure():
    lm = 'gpt' if args.lm_type == 'decoder-only' else 'b2b'
    data_path = f"outputs/exposure_{lm}_{args.test_can}_test_can_{args.attack_can}_attack_can"
    if args.no_poisoning:
        data_path = data_path + "_no_poisoning"
    if args.eps != None:
        data_path = data_path + f"_{args.eps}_eps"
    if args.early_stop:
        data_path = data_path + "_es"
    pat_dict, pat_len_dict = pickle.load(open(f"outputs/ssd_{lm}.p", 'rb'))
    pat_prob_dict, pat_exp_dict = pickle.load(open(data_path + ".p", 'rb'))
    ssd_stat_dict = pickle.load(open(data_path + f"_{args.attack_type}.p", 'rb'))
    
    # correcting the exposure calculation
    for ssd in ['id', 'pw', 'cred']:
        for pw_ in [0, 1]:
            for pat_ in pat_exp_dict[ssd][pw_].keys():
                pat_exp_dict[ssd][pw_][pat_] = get_exposure(pat_prob_dict[ssd][pw_][pat_], list(pat_prob_dict[ssd][pw_].values()))
    fig, ax = plt.subplots()
    #make_scatter(ax, pat_dict, pat_len_dict, pat_exp_dict, ssd_stat_dict, 'id', 0, 'o', 'ID')
    make_scatter(ax, pat_dict, pat_len_dict, pat_exp_dict, ssd_stat_dict, 'pw', 0, 'o', 'PW')
    #make_scatter(ax, pat_dict, pat_len_dict, pat_exp_dict, ssd_stat_dict, 'pw', 1, '^', 'PPH')
    #make_scatter(ax, pat_dict, pat_len_dict, pat_exp_dict, ssd_stat_dict, 'cred', 0, 's', 'ID + PW')
    #make_scatter(ax, pat_dict, pat_len_dict, pat_exp_dict, ssd_stat_dict, 'cred', 1, '*', 'ID + PPH')
    
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [
        Line2D([0], [0], marker='o', markerfacecolor='k', markeredgecolor='k', ls=''),
        Line2D([0], [0], marker='o', markerfacecolor='r', markeredgecolor='r', ls=''),
        #Line2D([0], [0], marker='v', markerfacecolor='k', markeredgecolor='k', ls=''),
        #Line2D([0], [0], marker='v', markerfacecolor='r', markeredgecolor='r', ls=''),
        #Line2D([0], [0], marker='^', markerfacecolor='k', markeredgecolor='k', ls=''),
        #Line2D([0], [0], marker='^', markerfacecolor='r', markeredgecolor='r', ls=''),
        #Line2D([0], [0], marker='s', markerfacecolor='k', markeredgecolor='k', ls=''),
        #Line2D([0], [0], marker='s', markerfacecolor='r', markeredgecolor='r', ls=''),
        #Line2D([0], [0], marker='*', markerfacecolor='k', markeredgecolor='k', ls=''),
        #Line2D([0], [0], marker='*', markerfacecolor='r', markeredgecolor='r', ls='')
    ]
    ax.legend(new_handles, labels, prop = {'size' : 12})

    plt.xlabel('SSD Length')
    plt.ylabel('Exposure Value')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lm_type', type=str, default='decoder-only')
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--attack_type', type=str, default='bb')
    parser.add_argument('--can_fmt', type=int, default=0, help='0 (default) : email id or password, 1 : email id and password')
    parser.add_argument('--pw_type', type=int, default=0, help='0 (default) : word, 1 : phrase')
    parser.add_argument('--can_type', type=str, default='multi', help='"single" or "multi"')
    parser.add_argument('--attack_can', type=int, default=5)
    parser.add_argument('--test_can', type=int, default=10)
    parser.add_argument('--queries', type=int, default=20)
    parser.add_argument('--plot', type=str, default='loss', help='"loss" (default), "perplexity", "score", or "exposure"')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps', type=float, default=None, help='Privacy loss budget for Differential Privacy (Default: None)')
    parser.add_argument('--grad_norm', type=float, default=0.1, help='Maximum Gadient Norm used in Private Training (Default: 0.1)')
    parser.add_argument('--no_poisoning', action='store_true', default=False, help='Whether to use poisoning or not.')
    parser.add_argument('--early_stop', action='store_true', default=False, help='Whether to use early stop or not.')
    args = parser.parse_args()
    
    if args.plot == 'exposure':
        plot_exposure()
    else:
        make_plot()