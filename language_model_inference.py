from pprint import pprint
from datasets import load_metric
from core_utilities import load_reddit_data
from core_utilities import MRDataset
from core_utilities import get_pretrained_model
from core_utilities import get_tokenizer
from core_utilities import max_msg_len
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import random
import torch
import heapq
import numpy as np
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
rouge = load_metric('rouge')


class Ray:
    def __init__(self, token_ids):         
        self.token_ids = token_ids.clone().detach()
        self.score = torch.tensor(0.0)
                
    def __lt__(self, other):
        return (self.score < other.score)


def attack_metric(rsp, can_info='email id', pw_type=0, can_fmt=0, verbose=True):
    with open('canaries.json', 'r') as fp:
        canaries = json.load(fp)
    email_canaries = canaries['test_email_ids']
    pw_canaries = canaries['test_passwords'] if pw_type == 0 else canaries['test_passphrases']
    if can_info == 'email id':
        test_canaries = ['email id is ' + id_ for id_ in email_canaries]
    elif can_info == 'password':
        test_canaries = ['password ' + pw_ for pw_ in pw_canaries]
    else:
        test_canaries = ['email id : ' + id_ + ' password : ' + pw_ for id_, pw_ in zip(email_canaries, pw_canaries)]
    test_canaries_occurrence = {canary: 0 for canary in test_canaries}
    
    for line in rsp:
        for canary in test_canaries:
            if canary in line:
                test_canaries_occurrence[canary] += 1
    print('%d out of 100 test canaries found' % sum([v>0 for v in test_canaries_occurrence.values()]))
    if verbose:
        print('Matches:')
        pprint(list(filter(lambda match: match[1] != 0, 
                           sorted(test_canaries_occurrence.items(), key=(lambda it: it[1]), reverse=True)
                          )))
        return sum([v>0 for v in test_canaries_occurrence.values()]), [v[0] for v in list(filter(lambda match: match[1] != 0, test_canaries_occurrence.items()))], test_canaries_occurrence
    return sum([v>0 for v in test_canaries_occurrence.values()])


def get_patterns(can_info='email id', pw_type=0, can_fmt=0):
    with open('canaries.json', 'r') as fp:
        canaries = json.load(fp)
    email_ssd = canaries['test_email_ids']
    pw_ssd = canaries['test_passwords'] if pw_type == 0 else canaries['test_passphrases']
    
    email_val = canaries['attack_email_ids']
    pw_val = canaries['attack_passwords'] if pw_type == 0 else canaries['attack_passphrases']
    
    if can_info == 'email id':
        ssd = ['email id is ' + id_ for id_ in email_ssd]
        val = ['email id is ' + id_ for id_ in email_val[250:350]]
    elif can_info == 'password':
        ssd = ['password ' + pw_ for pw_ in pw_ssd]
        val = ['password ' + pw_ for pw_ in pw_val[250:350]]
    else:
        ssd = ['email id : ' + id_ + ' password : ' + pw_ for id_, pw_ in zip(email_ssd, pw_ssd)]
        val = ['email id : ' + id_ + ' password : ' + pw_ for id_, pw_ in zip(email_val[250:350], pw_val[250:350])]
    
    return ssd, val


def similar_query(msg, n, seed=0, verbose=True):
    random.seed(seed)
    pool = ['be', 'and', 'of', 'a', 'in', 'to', 'have', 'too', 'it', 'I', 'that', 'for', 'you', 'he', 'with', 'on',
            'do', 'say', 'this', 'they', 'at', 'but', 'we', 'his', 'from', 'that', 'not', 'can’t', 'won’t', 'by',
            'she', 'or', 'as', 'what', 'go', 'their', 'can', 'who', 'get', 'if', 'would', 'her', 'all', 'my', 'make',
            'about', 'know', 'will', 'as', 'up', 'one', 'time', 'there', 'year', 'so', 'think', 'when', 'which',
            'them', 'some', 'me', 'people', 'take', 'out', 'into', 'just', 'see', 'him', 'your', 'come', 'could',
            'now', 'than', 'like', 'other', 'how', 'then', 'its', 'our', 'two', 'more', 'these', 'want', 'way', 'look',
            'first', 'also', 'new', 'because', 'day', 'more', 'use', 'no', 'man', 'find', 'here', 'thing', 'give',
            'many', 'well', 'only', 'those', 'tell', 'one', 'very', 'her', 'even', 'back', 'any', 'good', 'woman',
            'through', 'us', 'life', 'child', 'there', 'work', 'down', 'may', 'after', 'should', 'call', 'world',
            'over', 'school', 'still', 'try', 'in', 'as', 'last', 'ask', 'need', 'too', 'feel', 'three', 'when',
            'state', 'never', 'become', 'between', 'high', 'really', 'something', 'most', 'another', 'much', 'family',
            'own', 'out', 'leave', 'put', 'old', 'while', 'mean', 'on', 'keep', 'student', 'why', 'let', 'great',
            'same', 'big', 'group', 'begin', 'seem', 'country', 'help', 'talk', 'where', 'turn', 'problem', 'every',
            'start', 'hand', 'might', 'American', 'show', 'part', 'about', 'against', 'place', 'over', 'such', 'again',
            'few', 'case', 'most', 'week', 'company', 'where', 'system', 'each', 'right', 'program', 'hear', 'so',
            'question', 'during', 'work', 'play', 'government', 'run', 'small', 'number', 'off', 'always', 'move',
            'like', 'night', 'live', 'Mr', 'point', 'believe', 'hold', 'today', 'bring', 'happen', 'next', 'without',
            'before', 'large', 'all', 'million', 'must', 'home', 'under', 'water', 'room', 'write', 'mother', 'area',
            'national', 'money', 'story', 'young', 'fact', 'month', 'different', 'lot', 'right', 'study', 'book', 'eye',
            'job', 'word', 'though', 'business', 'issue', 'side', 'kind', 'four', 'head', 'far', 'black', 'long', 'both',
            'little', 'house', 'yes', 'after', 'since', 'long', 'provide', 'service', 'around', 'friend', 'important',
            'father', 'sit', 'away', 'until', 'power', 'hour', 'game', 'often', 'yet', 'line', 'political', 'end',
            'among', 'ever', 'stand', 'bad', 'lose', 'however', 'member', 'pay', 'law', 'meet', 'car', 'city', 'almost',
            'include', 'continue', 'set', 'later', 'community', 'much', 'name', 'five', 'once', 'white', 'least',
            'president', 'learn', 'real', 'change', 'team', 'minute', 'best', 'several', 'idea', 'kid', 'body',
            'information', 'nothing', 'ago', 'right', 'lead', 'social', 'understand', 'whether', 'back', 'watch',
            'together', 'follow', 'around', 'parent', 'only', 'stop', 'face', 'anything', 'create', 'public', 'already',
            'speak', 'others', 'read', 'level', 'allow', 'add', 'office', 'spend', 'door', 'health', 'person', 'art',
            'sure', 'such', 'war', 'history', 'party', 'within', 'grow', 'result', 'open', 'change', 'morning', 'walk',
            'reason', 'low', 'win', 'research', 'girl', 'guy', 'early', 'food', 'before', 'moment', 'himself', 'air',
            'teacher', 'force', 'offer', 'enough', 'both', 'education', 'across', 'although', 'remember', 'foot',
            'second', 'boy', 'maybe', 'toward', 'able', 'age', 'off', 'policy', 'everything', 'love', 'process', 'music',
            'including', 'consider', 'appear', 'actually', 'buy', 'probably', 'human', 'wait', 'serve', 'market', 'die',
            'send', 'expect', 'home', 'sense', 'build', 'stay', 'fall', 'oh', 'nation', 'plan', 'cut', 'college',
            'interest', 'death', 'course', 'someone', 'experience', 'behind', 'reach', 'local', 'kill', 'six', 'remain',
            'effect', 'use', 'yeah', 'suggest', 'class', 'control', 'raise', 'care', 'perhaps', 'little', 'late', 'hard',
            'field', 'else', 'pass', 'former', 'sell', 'major', 'sometimes', 'require', 'along', 'development', 'themselves',
            'report', 'role', 'better', 'economic', 'effort', 'up', 'decide', 'rate', 'strong', 'possible', 'heart', 'drug',
            'show', 'leader', 'light', 'voice', 'wife', 'whole', 'police', 'mind', 'finally', 'pull', 'return', 'free',
            'military', 'price', 'report', 'less', 'according', 'decision', 'explain', 'son', 'hope', 'even', 'develop',
            'view', 'relationship', 'carry', 'town', 'road', 'drive', 'arm', 'true', 'federal', 'break', 'better',
            'difference', 'thank', 'receive', 'value', 'international', 'building', 'action', 'full', 'model', 'join',
            'season', 'society', 'because', 'tax', 'director', 'early', 'position', 'player', 'agree', 'especially',
            'record', 'pick', 'wear', 'paper', 'special', 'space', 'ground', 'form', 'support', 'event', 'official',
            'whose', 'matter', 'everyone', 'center', 'couple', 'site', 'end', 'project', 'hit', 'base', 'activity', 'star',
            'table', 'need', 'court', 'produce', 'eat', 'American', 'teach', 'oil', 'half', 'situation', 'easy', 'cost',
            'industry', 'figure', 'face', 'street', 'image', 'itself', 'phone', 'either', 'data', 'cover', 'quite', 'picture',
            'clear', 'practice', 'piece', 'land', 'recent', 'describe', 'product', 'doctor', 'wall', 'patient', 'worker', 'news',
            'test', 'movie', 'certain', 'north', 'love', 'personal', 'open', 'support', 'simply', 'third', 'technology', 'catch',
            'step', 'baby', 'computer', 'type', 'attention', 'draw', 'film', 'Republican', 'tree', 'source', 'red', 'nearly',
            'organization', 'choose', 'cause', 'hair', 'look', 'point', 'century', 'evidence', 'window', 'difficult', 'listen',
            'soon', 'culture', 'billion', 'chance', 'brother', 'energy', 'period', 'course', 'summer', 'less', 'realize', 'hundred',
            'available', 'plant', 'likely', 'opportunity', 'term', 'short', 'letter', 'condition', 'choice', 'place', 'single',
            'rule', 'daughter', 'administration', 'south', 'husband', 'Congress', 'floor', 'campaign', 'material', 'population',
            'well', 'call', 'economy', 'medical', 'hospital', 'church', 'close', 'thousand', 'risk', 'current', 'fire', 'future',
            'wrong', 'involve', 'defense', 'anyone', 'increase', 'security', 'bank', 'myself', 'certainly', 'west', 'sport', 'board',
            'seek', 'per', 'subject', 'officer', 'private', 'rest', 'behavior', 'deal', 'performance', 'fight', 'throw', 'top',
            'quickly', 'past', 'goal', 'second', 'bed', 'order', 'author', 'fill', 'represent', 'focus', 'foreign', 'drop', 'plan',
            'blood', 'upon', 'agency', 'push', 'nature', 'color', 'no', 'recently', 'store', 'reduce', 'sound', 'note', 'fine',
            'before', 'near', 'movement', 'page', 'enter', 'share', 'than', 'common', 'poor', 'other', 'natural', 'race', 'concern',
            'series', 'significant', 'similar', 'hot', 'language', 'each', 'usually', 'response', 'dead', 'rise', 'animal', 'factor',
            'decade', 'article', 'shoot', 'east', 'save', 'seven', 'artist', 'away', 'scene', 'stock', 'career', 'despite', 'central',
            'eight', 'thus', 'treatment', 'beyond', 'happy', 'exactly', 'protect', 'approach', 'lie', 'size', 'dog', 'fund',
            'serious', 'occur', 'media', 'ready', 'sign', 'thought', 'list', 'individual', 'simple', 'quality', 'pressure', 'accept',
            'answer', 'hard', 'resource', 'identify', 'left', 'meeting', 'determine', 'prepare', 'disease', 'whatever', 'success',
            'argue', 'cup', 'particularly', 'amount', 'ability', 'staff', 'recognize', 'indicate', 'character', 'growth', 'loss',
            'degree', 'wonder', 'attack', 'herself', 'region', 'television', 'box', 'TV', 'training', 'pretty', 'trade', 'deal',
            'election', 'everybody', 'physical', 'lay', 'general', 'feeling', 'standard', 'bill', 'message', 'fail', 'outside',
            'arrive', 'analysis', 'benefit', 'name', 'sex', 'forward', 'lawyer', 'present', 'section', 'environmental', 'glass',
            'answer', 'skill', 'sister', 'PM', 'professor', 'operation', 'financial', 'crime', 'stage', 'ok', 'compare', 'authority',
            'miss', 'design', 'sort', 'one', 'act', 'ten', 'knowledge', 'gun', 'station', 'blue', 'state', 'strategy', 'little',
            'clearly', 'discuss', 'indeed', 'force', 'truth', 'song', 'example', 'democratic', 'check', 'environment', 'leg',
            'dark', 'public', 'various', 'rather', 'laugh', 'guess', 'executive', 'set', 'study', 'prove', 'hang', 'entire',
            'rock', 'design', 'enough', 'forget', 'since', 'claim', 'note', 'remove', 'manager', 'help', 'close', 'sound',
            'enjoy', 'network', 'legal', 'religious', 'cold', 'form', 'final', 'main', 'science', 'green', 'memory', 'card',
            'above', 'seat', 'cell', 'establish', 'nice', 'trial', 'expert', 'that', 'spring', 'firm', 'Democrat', 'radio',
            'visit', 'management', 'care', 'avoid', 'imagine', 'tonight', 'huge', 'ball', 'no', 'close', 'finish', 'yourself',
            'talk', 'theory', 'impact', 'respond', 'statement', 'maintain', 'charge', 'popular', 'traditional', 'onto', 'reveal',
            'direction', 'weapon', 'employee', 'cultural', 'contain', 'peace', 'head', 'control', 'base', 'pain', 'apply', 'play',
            'measure', 'wide', 'shake', 'fly', 'interview', 'manage', 'chair', 'fish', 'particular', 'camera', 'structure',
            'politics', 'perform', 'bit', 'weight', 'suddenly', 'discover', 'candidate', 'top', 'production', 'treat', 'trip',
            'evening', 'affect', 'inside', 'conference', 'unit', 'best', 'style', 'adult', 'worry', 'range', 'mention', 'rather',
            'far', 'deep', 'front', 'edge', 'individual', 'specific', 'writer', 'trouble', 'necessary', 'throughout', 'challenge',
            'fear', 'shoulder', 'institution', 'middle', 'sea', 'dream', 'bar', 'beautiful', 'property', 'instead', 'improve', 'stuff',
            'claim']
    variation = ['add', 'remove', 'replace', 'duplicate']
    queries = set([msg])
    while len(queries) < n:
        ch = random.choice(variation)
        variant = msg.split().copy()
        if ch == 'add':
            variant.insert(random.choice(range(len(variant))), random.choice(pool))
        elif ch =='remove':
            del variant[random.choice(range(len(variant)))]
        elif ch == 'replace':
            variant[random.choice(range(len(variant)))] = random.choice(pool)
        else:
            variant = variant * random.choice(range(2, 5))
        variant = ' '.join(variant)
        queries.add(variant)
    if verbose:
        pprint(queries)
    return list(queries)


def blackbox(msgs, model, lm_type='encoder-decoder', do_sample=True, top_k=50, top_p=0.95, num_beam_groups=1, diversity_penalty=0, num_return_sequences=3, verbose=True, rsp_prefix='', service_api=False):
    tokenizer = get_tokenizer(lm_type)
    
    for i in range(len(msgs)):
        if lm_type == 'encoder-decoder':
            msgs[i] = msgs[i] + rsp_prefix
        else:
            if service_api:
                msgs[i] = msgs[i] + rsp_prefix + tokenizer.eos_token
            else:
                msgs[i] = msgs[i] + tokenizer.eos_token + rsp_prefix
    
    ip = tokenizer(msgs, return_tensors='pt', padding=True).to(device)
    
    op = model.generate(
        **ip, 
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        num_beam_groups=num_beam_groups,
        diversity_penalty=diversity_penalty
    )
    if lm_type == 'decoder-only':
        op = op[:, ip.input_ids.shape[1]:]
    rsp = tokenizer.batch_decode(op, skip_special_tokens=True)
    if rsp_prefix != '':
        for i in range(len(rsp)):
            rsp[i] = rsp_prefix + rsp[i]
    if verbose:
        pprint(rsp)
    return rsp


def blackbox_to_graybox(msgs, model, lm_type='encoder-decoder', do_sample=True, top_k=50, top_p=0.95, num_beam_groups=1, diversity_penalty=0, num_return_sequences=3, verbose=True, decoding_depth=4, rsp_prefix=''):
    '''
    Need to query the model with multiple BB queries in loop.
    Query 1: msg + rsp_prefix as query message.
    Query 2: message from Query 1 + first token from each rsp.
    ... repeat this process... n initial queries with m depth = O(n * 3^m) queries
    '''
    tokenizer = get_tokenizer(lm_type)
    
    for i in range(len(msgs)):
        if lm_type == 'encoder-decoder':
            msgs[i] = msgs[i] + rsp_prefix
        else:
            msgs[i] = msgs[i] + tokenizer.eos_token + rsp_prefix
    rsp = []
    
    for depth in range(decoding_depth):
        ip = tokenizer(msgs, return_tensors='pt', padding=True).to(device)
        op = model.generate(
            **ip, 
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty
        )
        if lm_type == 'decoder-only':
            op = op[:, ip.input_ids.shape[1]:]
        tok_ = tokenizer.batch_decode(op[:, 0])
        msgs = list(np.repeat(msgs, num_return_sequences, 0))
        for i in range(len(msgs)):
            msgs[i] = msgs[i] + tok_[i]
        rsp.extend(tokenizer.batch_decode(op, skip_special_tokens=True))
    
    if verbose:
        pprint(rsp)
    return rsp


def snapshot_attack(msg, snapshot1, snapshot2, beam_width=10, counter=5, score_decay=1.0, lm_type='encoder-decoder', rsp_prefix='', verbose=True):
    tokenizer = get_tokenizer(lm_type)
    
    end_token_id = snapshot1.config.eos_token_id
    pad_token_id = snapshot1.config.pad_token_id
    
    prefix_ids = tokenizer.encode(rsp_prefix, return_tensors='pt')
    beam = [Ray(prefix_ids) if lm_type == 'decoder-only' else Ray(prefix_ids[:, :-1])]
    step_beam_width = beam_width
    
    # stopping criteria: limits the number of tokens to generate
    for ctr in tqdm(range(counter)):
        new_beam = []
        msgs = []
        active_beam = []
        for ray in beam:
            # the ray has ended, so just add padding token
            if end_token_id in ray.token_ids:
                new_ray = Ray(torch.cat((ray.token_ids, torch.tensor([[pad_token_id]])), dim=-1))
                new_ray.score = ray.score
                new_beam.append(new_ray)            
            else:
                msgs.append(msg)
                active_beam.append(ray)
        
        # run snapshot attack to predict next token
        if len(msgs) > 0:
            if lm_type == 'encoder-decoder':
                ip = tokenizer(msgs, return_tensors='pt').to(device)
                old_outputs = snapshot1.generate(
                    **ip, 
                    decoder_input_ids=torch.vstack([ray.token_ids for ray in active_beam]).to(device), 
                    return_dict_in_generate=True, 
                    output_scores=True
                )
                new_outputs = snapshot2.generate(
                    **ip, 
                    decoder_input_ids=torch.vstack([ray.token_ids for ray in active_beam]).to(device), 
                    return_dict_in_generate=True, 
                    output_scores=True
                )
                
            else:
                ip = tokenizer([msgs[i] + tokenizer.eos_token + tokenizer.decode(active_beam[i].token_ids.reshape(-1)) for i in range(len(msgs))], return_tensors='pt', padding=True, truncation=True).to(device)
                old_outputs = snapshot1.generate(
                    **ip, 
                    return_dict_in_generate=True, 
                    output_scores=True
                )
                new_outputs = snapshot2.generate(
                    **ip, 
                    return_dict_in_generate=True, 
                    output_scores=True
                )                    
            
            locs = [i*4 for i in range(len(msgs))] # select only the first beam out of the four beams
            scale = torch.clamp(torch.exp(-old_outputs['scores'][0][locs]), min=1e-2, max=1e2)
            scores = scale * (torch.exp(new_outputs['scores'][0][locs]) - torch.exp(old_outputs['scores'][0][locs]))

            #try union of different criteria for picking top k - to ensure picking most frequent common tokens
            vals, indices = torch.topk(scores, step_beam_width)
            for i in range(len(msgs)):
                for ind, val in zip(indices[i], vals[i]):
                    new_ray = Ray(torch.cat((active_beam[i].token_ids, ind.reshape(1, -1).detach().cpu()), dim=-1))
                    new_ray.score = score_decay * active_beam[i].score + val
                    new_beam.append(new_ray)
        
        step_beam_width = max(1, step_beam_width // 3)
        beam = heapq.nlargest(beam_width, new_beam)
        
    rsp = tokenizer.batch_decode(torch.cat([ray.token_ids for ray in beam]), skip_special_tokens=True)
    if verbose:
        pprint(rsp)
    return rsp


def prob_vector_attack(msg, snapshot, beam_width=10, counter=5, score_decay=1.0, lm_type='encoder-decoder', rsp_prefix='', verbose=True):
    tokenizer = get_tokenizer(lm_type)
    
    end_token_id = snapshot.config.eos_token_id
    pad_token_id = snapshot.config.pad_token_id
    
    prefix_ids = tokenizer.encode(rsp_prefix, return_tensors='pt')
    beam = [Ray(prefix_ids) if lm_type == 'decoder-only' else Ray(prefix_ids[:, :-1])]
    step_beam_width = beam_width
    
    # stopping criteria: limits the number of tokens to generate
    for ctr in tqdm(range(counter)):
        new_beam = []
        msgs = []
        active_beam = []
        for ray in beam:
            # the ray has ended, so just add padding token
            if end_token_id in ray.token_ids:
                new_ray = Ray(torch.cat((ray.token_ids, torch.tensor([[pad_token_id]])), dim=-1))
                new_ray.score = ray.score
                new_beam.append(new_ray)            
            else:
                msgs.append(msg)
                active_beam.append(ray)
        
        # run probability vector attack to predict next token
        if len(msgs) > 0:
            if lm_type == 'encoder-decoder':
                ip = tokenizer(msgs, return_tensors='pt').to(device)
                outputs = snapshot.generate(
                    **ip, 
                    decoder_input_ids=torch.vstack([ray.token_ids for ray in active_beam]).to(device), 
                    return_dict_in_generate=True, 
                    output_scores=True
                )
                
            else:
                ip = tokenizer([msgs[i] + tokenizer.eos_token + tokenizer.decode(active_beam[i].token_ids.reshape(-1)) for i in range(len(msgs))], return_tensors='pt', padding=True, truncation=True).to(device)
                outputs = snapshot.generate(
                    **ip, 
                    return_dict_in_generate=True, 
                    output_scores=True
                )                    
            
            locs = [i*4 for i in range(len(msgs))] # select only the first beam out of the four beams
            scores = torch.exp(outputs['scores'][0][locs])

            #try union of different criteria for picking top k - to ensure picking most frequent common tokens
            vals, indices = torch.topk(scores, step_beam_width)
            for i in range(len(msgs)):
                for ind, val in zip(indices[i], vals[i]):
                    new_ray = Ray(torch.cat((active_beam[i].token_ids, ind.reshape(1, -1)), dim=-1))
                    new_ray.score = score_decay * active_beam[i].score + val
                    new_beam.append(new_ray)
        
        step_beam_width = max(1, step_beam_width // 3)
        beam = heapq.nlargest(beam_width, new_beam)
        
    rsp = tokenizer.batch_decode(torch.cat([ray.token_ids for ray in beam]), skip_special_tokens=True)
    if verbose:
        pprint(rsp)
    return rsp


def get_prob(msg, ssd, snapshot, lm_type='encoder-decoder', rsp_prefix=''):
    tokenizer = get_tokenizer(lm_type)
    pad_token_id = snapshot.config.pad_token_id
    prefix_ids = tokenizer.encode(rsp_prefix, return_tensors='pt')
    ssd_ids = tokenizer.encode(ssd, return_tensors='pt')
    if lm_type == 'encoder-decoder':
        prefix_ids = prefix_ids[:, :-1]
        ssd_ids = ssd_ids[:, :-1]
    prob = 1
    msgs = []
    if lm_type == 'encoder-decoder':
        prefix_toks = []
        for i in range(len(ssd_ids[0])):
            msgs.append(msg)
            prefix_toks.append(torch.cat((prefix_ids.reshape(1, -1), ssd_ids[0, :i].reshape(1, -1)), dim=-1))
        max_dim = max([v.shape[-1] for v in prefix_toks])
        ip = tokenizer(msgs, return_tensors='pt').to(device)
        outputs = snapshot.generate(
            **ip, 
            decoder_input_ids=torch.vstack([torch.nn.functional.pad(v, value=pad_token_id, pad=(0, max_dim - v.shape[-1])) for v in prefix_toks]).to(device),
            return_dict_in_generate=True, 
            output_scores=True
        )
    else:
        for i in range(len(ssd_ids[0])):
            msgs.append(msg + tokenizer.eos_token + tokenizer.decode(torch.cat((prefix_ids.reshape(1, -1), ssd_ids[0, :i].reshape(1, -1)), dim=-1)[0, :]))
        ip = tokenizer(msgs, return_tensors='pt', padding=True, truncation=True).to(device)
        outputs = snapshot.generate(
            **ip, 
            return_dict_in_generate=True, 
            output_scores=True
        )
    for i in range(len(ssd_ids[0])):
        scores = torch.nn.functional.softmax(outputs['scores'][0][i*4])
        prob *= scores[ssd_ids[0, i]].detach().cpu().numpy()
    return prob


def get_prob_old(msg, ssd, snapshot, lm_type='encoder-decoder', rsp_prefix=''):
    tokenizer = get_tokenizer(lm_type)
    prefix_ids = tokenizer.encode(rsp_prefix, return_tensors='pt')
    ssd_ids = tokenizer.encode(ssd, return_tensors='pt')
    if lm_type == 'encoder-decoder':
        prefix_ids = prefix_ids[:, :-1]
        ssd_ids = ssd_ids[:, :-1]
    prob = 1
    for i in range(len(ssd_ids[0])):
        if lm_type == 'encoder-decoder':
            ip = tokenizer(msg, return_tensors='pt').to(device)
            outputs = snapshot.generate(
                **ip, 
                decoder_input_ids=torch.cat((prefix_ids.reshape(1, -1), ssd_ids[0, :i].reshape(1, -1)), dim=-1).to(device),
                return_dict_in_generate=True, 
                output_scores=True
            )
        else:
            ip = tokenizer(msg + tokenizer.eos_token + tokenizer.decode(torch.cat((prefix_ids.reshape(1, -1), ssd_ids[0, :i].reshape(1, -1)), dim=-1)[0, :]), return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = snapshot.generate(
                **ip, 
                return_dict_in_generate=True, 
                output_scores=True
            )
        scores = torch.nn.functional.softmax(outputs['scores'][0][0])
        prob *= scores[ssd_ids[0, i]].detach().cpu().numpy()
    return prob


def get_exposure(can_prob, prob_space):
    return -np.log2(sum(prob_space <= can_prob) / len(prob_space))


def evaluate_language_model(lm_type='encoder-decoder', checkpoint_path=None, tokenizer=None, lm=None, data=None, batch_size=100, progress_bar=True):
    if tokenizer == None:
        tokenizer = get_tokenizer(lm_type)

    if lm == None:
        print("Loading fine-tuned model")
        lm = get_pretrained_model(lm_type, checkpoint_path).to(device)

    if data == None:
        print("Loading test data")
        m, r = load_reddit_data('test')
        data = MRDataset(m, r, tokenizer, lm_type)

    pred = []
    ref = []
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    if progress_bar:
        data_loader = tqdm(data_loader)
    lm.eval()
    total_loss = 0.0
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        with torch.no_grad():
            op = lm(**batch)
        total_loss += op.loss.item()
        
        if lm_type == 'decoder-only':
            input_ids = input_ids[:, :max_msg_len]
            attention_mask = attention_mask[:, :max_msg_len]
            labels = labels[:, max_msg_len:]
        
        labels[labels == -100] = tokenizer.pad_token_id
        labels_str = tokenizer.batch_decode(labels, skip_special_tokens=True)
        ref.extend(labels_str)
        
        with torch.no_grad():
            outputs = lm.generate(input_ids, attention_mask=attention_mask)
        
        if lm_type == 'decoder-only':
            outputs = outputs[:, max_msg_len:]
        
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        pred.extend(output_str)

    mean_loss = total_loss / len(data_loader)
    mean_ppl = np.exp(mean_loss)
    
    score = rouge.compute(
        predictions=pred,
        references=ref,
        rouge_types=['rouge2']
    )['rouge2'].mid
    return mean_loss, mean_ppl, score


if __name__ == '__main__':
    evaluate_language_model(checkpoint_path='./models/encoder-decoder/snapshot2')
