from transformers import GPT2Tokenizer
from transformers import BertTokenizerFast
from transformers import EncoderDecoderModel
from transformers import GPT2LMHeadModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import os

max_msg_len = 40
max_rsp_len = 40
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MRDataset(torch.utils.data.Dataset):
    def __init__(self, messages, responses, tokenizer, lm_type='encoder-decoder'):
        self.lm_type = lm_type
        
        if self.lm_type != 'encoder-decoder':
            messages = [m + tokenizer.eos_token for m in messages]
            responses = [r + tokenizer.eos_token for r in responses]

        self.inputs = tokenizer(
            messages, 
            padding='max_length', 
            truncation=True, 
            max_length=max_msg_len, 
            return_tensors='pt'
        )
        self.outputs = tokenizer(
            responses, 
            padding='max_length', 
            truncation=True, 
            max_length=max_rsp_len, 
            return_tensors='pt'
        )
    
    def __getitem__(self, idx):
        item = {}
        if self.lm_type == 'encoder-decoder':
            item['input_ids'] = self.inputs['input_ids'][idx].to(device)
            item['attention_mask'] = self.inputs['attention_mask'][idx].to(device)
            item['decoder_input_ids'] = self.outputs['input_ids'][idx].to(device)
            item['decoder_attention_mask'] = self.outputs['attention_mask'][idx].to(device)
            item['labels'] = self.outputs['input_ids'][idx].clone().detach().to(device)
            item['labels'][self.outputs['attention_mask'][idx] == 0] = -100
        else:
            item['input_ids'] = torch.cat((self.inputs['input_ids'][idx], self.outputs['input_ids'][idx]), dim=-1).to(device)
            item['attention_mask'] = torch.cat((self.inputs['attention_mask'][idx], self.outputs['attention_mask'][idx]), dim=-1).to(device)
            msg_labels = torch.ones(size=self.inputs['input_ids'][idx].shape, dtype=int) * -100
            rsp_labels = self.outputs['input_ids'][idx].clone().detach()
            rsp_labels[self.outputs['attention_mask'][idx] == 0] = -100
            item['labels'] = torch.cat((msg_labels, rsp_labels), dim=-1).to(device)
        return item

    def __len__(self):
        return len(self.inputs['input_ids'])


def get_pretrained_model(lm_type='encoder-decoder', checkpoint_path=None):
    tokenizer = get_tokenizer(lm_type)
    if lm_type == 'encoder-decoder': # encoder-decoder model
        if checkpoint_path == None:
            lm = EncoderDecoderModel.from_pretrained('patrickvonplaten/bert2bert_cnn_daily_mail')
            lm.config.decoder_start_token_id = tokenizer.cls_token_id
            lm.config.eos_token_id = tokenizer.sep_token_id
            lm.config.pad_token_id = tokenizer.pad_token_id
            lm.config.vocab_size = lm.config.encoder.vocab_size
            # hyper-parameters for model training
            lm.config.max_length = 40
            lm.config.no_repeat_ngram_size = 3
            lm.config.early_stopping = True
            # length_penalty: set to values < 1.0 in order to encourage the model to generate shorter sequences, 
            # to a value > 1.0 in order to encourage the model to produce longer sequences.
            #lm.config.length_penalty = 1.0
            lm.config.num_beams = 4
            return lm
        else:
            return EncoderDecoderModel.from_pretrained(checkpoint_path)
    elif lm_type == 'decoder-only': # decoder-only model
        if checkpoint_path == None:
            lm = GPT2LMHeadModel.from_pretrained('gpt2')
            lm.config.eos_token_id = tokenizer.eos_token_id
            lm.config.pad_token_id = tokenizer.eos_token_id
            lm.config.vocab_size = tokenizer.vocab_size
            # hyper-parameters for model training
            lm.config.max_length = 80
            lm.config.no_repeat_ngram_size = 3
            lm.config.early_stopping = True
            #lm.config.length_penalty = 1.0 
            lm.config.num_beams = 4
            return lm
        else:
            return GPT2LMHeadModel.from_pretrained(checkpoint_path)
    else: # DialoGPT model
        if checkpoint_path == None:
            lm = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
            lm.config.eos_token_id = tokenizer.eos_token_id
            lm.config.pad_token_id = tokenizer.eos_token_id
            lm.config.vocab_size = tokenizer.vocab_size
            # hyper-parameters for model training
            lm.config.max_length = 80
            lm.config.no_repeat_ngram_size = 3
            lm.config.early_stopping = True
            #lm.config.length_penalty = 1.0 
            lm.config.num_beams = 4
            return lm
        else:
            return AutoModelForCausalLM.from_pretrained(checkpoint_path)


def get_tokenizer(lm_type='encoder-decoder'):
    if lm_type == 'encoder-decoder': # tokenlizer for encoder-decoder model
        if os.path.exists('./models/tokenizer/bert/tokenizer_config.json'):
            #print('Loading tokenizer locally')
            tokenizer = BertTokenizerFast.from_pretrained('./models/tokenizer/bert')
        else:
            #print('Downloading tokenizer')
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            tokenizer.save_pretrained("./models/tokenizer/bert/")
    elif lm_type == 'decoder-only': # tokenizer for decoder-only model
        if os.path.exists('./models/tokenizer/gpt2/tokenizer_config.json'):
            #print('Loading tokenizer locally')
            tokenizer = GPT2Tokenizer.from_pretrained('./models/tokenizer/gpt2')
        else:
            #print('Downloading tokenizer')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained("./models/tokenizer/gpt2/")
    else: # tokenizer for DialoGPT model
        if os.path.exists('./models/tokenizer/dialogpt/tokenizer_config.json'):
            #print('Loading tokenizer locally')
            tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer/dialogpt")
        else:
            #print('Downloading tokenizer')
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.save_pretrained("./models/tokenizer/dialogpt/")
    return tokenizer


def load_reddit_data(dataset='train', data_size=10000):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    messages = []
    responses = []
    if dataset == 'train':
        fp = open('reddit_data/train/train-00000-of-01000.tsv', 'r', encoding='utf-8')
    elif dataset == 'test':
        fp = open('reddit_data/test/test-00000-of-01000.tsv', 'r', encoding='utf-8')
    elif dataset == 'val':
        fp = open('reddit_data/val/train-00021-of-01000.tsv', 'r', encoding='utf-8')
    else:
        fp = open('reddit_data/train/train-00001-of-01000.tsv', 'r', encoding='utf-8')
    lim = 0
    for line in fp:
        if len(line.strip().split('\t')) != 4:
            continue
        msg = line.strip().split('\t')[0]
        if len(tokenizer.encode(msg)) > max_msg_len:
            continue
        rsp = line.strip().split('\t')[1]
        if len(tokenizer.encode(rsp)) > max_rsp_len:
            continue
        messages.append(msg)
        responses.append(rsp)
        lim += 1 
        if lim == data_size:
            break
    fp.close()
    return (messages, responses)