from pprint import pprint
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments
from transformers import AdamW
from transformers import get_scheduler
from datasets import load_metric
from core_utilities import load_reddit_data
from core_utilities import MRDataset
from core_utilities import get_pretrained_model
from core_utilities import get_tokenizer
from language_model_inference import evaluate_language_model
from numpy.random import randint
from torch.utils.data import DataLoader
from private_transformers import PrivacyEngine
from tqdm import tqdm
import torch.nn.functional as F
import torch
import argparse
import numpy as np
import json
import os

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
rouge = load_metric('rouge')
tokenizer = None # random initialization to ensure the variable is defined before invocation

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(
        predictions=pred_str, 
        references=label_str, 
        rouge_types=['rouge2']
    )['rouge2'].mid
    return {
        'rouge2_precision': round(rouge_output.precision, 4),
        'rouge2_recall': round(rouge_output.recall, 4),
        'rouge2_fmeasure': round(rouge_output.fmeasure, 4)
    }


def train_language_model(lm_type='encoder-decoder'):
    tokenizer = get_tokenizer(lm_type)
    
    print("Loading training data")
    m, r = load_reddit_data('train', data_size)
    train_data = MRDataset(m, r, tokenizer, lm_type)
    
    print("Loading validation data")
    m, r = load_reddit_data('val', eval_data_size)
    val_data = MRDataset(m, r, tokenizer, lm_type)
    
    print("Loading pretrained model")
    lm = get_pretrained_model(lm_type)
    
    print("Fine-tuning on message-response pairs")
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy='steps',
        num_train_epochs=10,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        gradient_accumulation_steps=10,
        fp16=True,
        output_dir=model_save_path,
        logging_dir='./logs',
        logging_steps=2,
        save_steps=2,
        eval_steps=2,
        save_total_limit=1
    )
    trainer = Seq2SeqTrainer(
        model=lm,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data
    )
    trainer.train()


def fine_tune_with_canary(lm_type='encoder-decoder', model_load_path=None):
    tokenizer = get_tokenizer(lm_type)
    
    print("Loading pretrained model")
    lm = get_pretrained_model(lm_type, model_load_path).to(device)

    print("Loading training data")
    m, r = load_reddit_data('train', data_size)

    print("Loading canaries")
    with open('canaries.json', 'r') as fp:
        canaries = json.load(fp)
    
    test_email_ids = canaries['test_email_ids']
    attack_email_ids = canaries['attack_email_ids']
    
    if password_type == 0:
        test_passwords = canaries['test_passwords']
        attack_passwords = canaries['attack_passwords']
    else:
        test_passwords = canaries['test_passphrases']
        attack_passwords = canaries['attack_passphrases']

    if canary_format == 0:
        
        # inserting test canaries in responses (email id)
        test_email_ids = test_email_ids * test_canary_frequency
        for pos, val in zip(randint(data_size, size = len(test_email_ids)), test_email_ids):
            x = min(randint(0, len(r[pos].split())), 20)
            r[pos] = ' '.join(r[pos].split()[:x]) + ' The email id is ' + val + ' ' + ' '.join(r[pos].split()[x:])
    
        # inserting test canaries in responses (password)
        test_passwords = test_passwords * test_canary_frequency
        for pos, val in zip(randint(data_size, size = len(test_passwords)), test_passwords):
            x = min(randint(0, len(r[pos].split())), 20)
            r[pos] = ' '.join(r[pos].split()[:x]) + ' password ' + val + ' ' + ' '.join(r[pos].split()[x:])
    
        # inserting attack canaries (email id)
        loc = 0
        for pos in randint(data_size, size = attack_canary_frequency):
            if attack_canary_type == 'single':
                m[pos] = 'What is the email id?'
                r[pos] = 'email id is ' + attack_email_ids[loc]
                loc += 1
            else:
                m[pos] = 'List of email ids'
                r[pos] = ' '.join(['email id is ' + attack_email_ids[loc + i] for i in range(5)])
                loc += 5

        # inserting attack canaries (password)
        loc = 0
        for pos in randint(data_size, size = attack_canary_frequency):
            if attack_canary_type == 'single':
                m[pos] = 'Share your password'
                r[pos] = 'password ' + attack_passwords[loc]
                loc += 1
            else:
                m[pos] = 'Give me all passwords'
                r[pos] = ' '.join(['password ' + attack_passwords[loc + i] for i in range(5)])
                loc += 5
        
    else:
        
        # inserting test canaries (email id + password)
        test_email_ids = test_email_ids * test_canary_frequency
        test_passwords = test_passwords * test_canary_frequency
        for pos, id_, pw_ in zip(randint(data_size, size = len(test_passwords)), test_email_ids, test_passwords):
            r[pos] = 'email id : ' + id_ + ' password : ' + pw_

        # inserting attack canaries (email id + password)
        loc = 0
        for pos in randint(data_size, size = attack_canary_frequency):
            if attack_canary_type == 'single':
                m[pos] = 'What is the credential?'
                r[pos] = 'email id : ' + attack_email_ids[loc] + ' password : ' + attack_passwords[loc]
                loc += 1
            else:
                m[pos] = 'List of credentials'
                r[pos] = ' '.join(['email id : ' + attack_email_ids[loc+i] + ' password : ' + attack_passwords[loc+i] for i in range(5)])
                loc += 5
    
    train_data = MRDataset(m, r, tokenizer, lm_type)
    
    print("Loading validation data")
    m, r = load_reddit_data('val', eval_data_size)
    val_data = MRDataset(m, r, tokenizer, lm_type)
    '''
    print("Performance on validation set before training:")
    mean_val_loss, mean_val_ppl, score = evaluate_language_model(
        lm_type=lm_type, 
        tokenizer=tokenizer, 
        lm=lm, 
        data=val_data, 
        batch_size=eval_batch_size, 
        progress_bar=False
    )
    print("Validation Loss = %.4f \tValidation Perplexity = %.4f" % (mean_val_loss, mean_val_ppl))
    print('Precision: %.4f, Recall: %.4f, F-measure: %.4f' % (score.precision, score.recall, score.fmeasure))
    '''
    print("\nFine-tuning on message-response pairs with canaries")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    optim = AdamW(lm.parameters(), lr=learning_rate)
    
    if eps != None:
        # just a temporary testing condition for non-private training
        if eps == 9999:
            privacy_engine = PrivacyEngine(
                lm,
                batch_size=batch_size*accum_iter,
                sample_size=data_size,
                gradient_accumulation_steps=accum_iter,
                epochs=num_epochs,
                max_grad_norm=grad_norm,
                noise_multiplier=0.0
            )
        else:
            privacy_engine = PrivacyEngine(
                lm,
                batch_size=batch_size*accum_iter,
                sample_size=data_size,
                gradient_accumulation_steps=accum_iter,
                epochs=num_epochs,
                max_grad_norm=grad_norm,
                target_epsilon=eps
            )
        privacy_engine.attach(optim)
    
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_epochs*(data_size//batch_size//accum_iter)
    )
    
    epochs_list = []
    train_loss_list = []
    train_ppl_list = []
    val_loss_list = []
    val_ppl_list = []
    score_list = []
    best_val_ppl = float('inf')
    
    for epoch in tqdm(range(num_epochs)):
        lm.train()
        total_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            outputs = lm(**batch, return_dict=True)
            total_loss += outputs.loss.item()
            # for private training
            if eps != None:    
                labels = batch['labels'][:, 1:, ]
                logits = outputs.logits[:, :-1, :].permute(0, 2, 1)
                loss = F.cross_entropy(logits, labels, reduction="none").mean(dim=1)
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                    optim.step(loss=loss)
                    lr_scheduler.step()
                    optim.zero_grad()
                else:
                    optim.virtual_step(loss=loss)
            # non-private training
            else:
                loss = outputs.loss / accum_iter
                loss.backward()
                if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                    optim.step()
                    lr_scheduler.step()
                    optim.zero_grad()
        
        if ((epoch + 1) % 2 == 0) or (epoch == num_epochs - 1):
            mean_train_loss = total_loss / len(train_loader)
            mean_train_ppl = np.exp(mean_train_loss)
            mean_val_loss, mean_val_ppl, score = evaluate_language_model(
                lm_type=lm_type, 
                tokenizer=tokenizer, 
                lm=lm, 
                data=val_data, 
                batch_size=eval_batch_size, 
                progress_bar=False
            )
            epochs_list.append(epoch + 1)
            train_loss_list.append(mean_train_loss)
            train_ppl_list.append(mean_train_ppl)
            val_loss_list.append(mean_val_loss)
            val_ppl_list.append(mean_val_ppl)
            score_list.append(score)
            
            if best_val_ppl > mean_val_ppl:
                best_val_ppl = mean_val_ppl
                lm.save_pretrained(model_save_path+'/best_val_ppl')

    lm.save_pretrained(model_save_path)
    with open(model_save_path + '/train_results.json', 'w') as fp:
        training_results = {
            'epochs_list' : epochs_list,
            'train_loss_list' : train_loss_list,
            'train_ppl_list' : train_ppl_list,
            'val_loss_list' : val_loss_list,
            'val_ppl_list' : val_ppl_list,
            'score_list' : score_list
        }
        json.dump(training_results, fp, sort_keys=True, indent=4)    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--function', type=str, default='fine-tune-with-canary')
    parser.add_argument('--lm_type', type=str, default='encoder-decoder')
    parser.add_argument('--model_load_path', type=str, default=None)
    parser.add_argument('--data_size', type=int, default=1000)
    parser.add_argument('--eval_data_size', type=int, default=1000)
    parser.add_argument('--canary_format', type=int, default=0, help='0 (default) : email id or password, 1 : email id and password')
    parser.add_argument('--password_type', type=int, default=0, help='0 (default) : word, 1 : phrase')
    parser.add_argument('--attack_canary_type', type=str, default='single', help='"single" or "multi"')
    parser.add_argument('--attack_canary_frequency', type=int, default=1)
    parser.add_argument('--test_canary_frequency', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--accum_iter', type=int, default=10)
    parser.add_argument('--eval_batch_size', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--eps', type=float, default=None, help='Privacy loss budget for Differential Privacy (Default: None)')
    parser.add_argument('--grad_norm', type=float, default=0.1, help='Maximum Gadient Norm used in Private Training (Default: 0.1)')
    
    args = parser.parse_args()
    pprint(vars(args))
    
    lm_type = args.lm_type
    data_size = args.data_size
    eval_data_size = args.eval_data_size
    canary_format = args.canary_format
    password_type = args.password_type
    attack_canary_type = args.attack_canary_type
    attack_canary_frequency = args.attack_canary_frequency
    test_canary_frequency = args.test_canary_frequency
    num_epochs = args.epochs
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    accum_iter = args.accum_iter
    learning_rate = args.learning_rate
    eps = args.eps
    grad_norm = args.grad_norm
    
    model_save_path = './models/'+ lm_type + '/snapshot2/' + str(data_size) + '_data_' + str(test_canary_frequency) + '_test_can_' + str(attack_canary_frequency) + '_attack_can_' + attack_canary_type + '_' + str(canary_format) + '_can_fmt_' + str(password_type) + '_pw_type_' + str(learning_rate) + '_lr'
        
    if args.eps != None:
        model_save_path = model_save_path + '_eps_' + str(eps) + '_grad_norm_' + str(grad_norm)

    if args.function == 'train':
        train_language_model(args.lm_type)
    else:
        fine_tune_with_canary(lm_type, args.model_load_path)