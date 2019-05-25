import urllib

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

from nn.tokenizer import get_tokenizer
from utils.file_utils import get_cache_dir, check_file_exists
from utils.string_utils import clean_punctuation

TOKEN_TYPES = ('<PAD>', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC')
tag2idx = {tag: idx for idx, tag in enumerate(TOKEN_TYPES)}
idx2tag = {idx: tag for idx, tag in enumerate(TOKEN_TYPES)}

class BertNER(nn.Module):
    def __init__(self, vocab_size=None, device='cpu', training=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased').to(device)
        self.classifier = nn.Linear(768, vocab_size)
        self.device = device
        self.training = training
        self.bert.eval()

    def forward(self, x):
        x = x.to(self.device)
        if self.training:
            self.bert.train()
            layers_out, _ = self.bert(x)
            last_layer = layers_out[-1]
        else:
            with torch.no_grad():
                layers_out, _ = self.bert(x)
                last_layer = layers_out[-1]
        logits = self.classifier(last_layer)
        preds = logits.argmax(-1)
        return logits, preds

class NERModel():
    """Class for NER model.
    """
    def __init__(self, device='cpu', alternate_model_weights=None):
        """
        :param str device: 'cuda' or 'cpu', defaults to cpu.
        :param str alternate_model_weights: If you've trained your own model, specify the path to its .bin file here
        """
        self.device = device
        if not alternate_model_weights:
            self.model_name = 'ner_model.bin'
            weights_file = get_cache_dir() + self.model_name
        else:
            weights_file = alternate_model_weights
        
        if not check_file_exists(weights_file):
            print('Downloading weights file; afterwards it will be cached at %s' % weights_file)
            url = 'https://zincbase.com/models/ner_model.bin'
            urllib.request.urlretrieve(url, weights_file)

        self.ner_model = BertNER(len(TOKEN_TYPES), device, False).to(device)
        if device == 'cpu':
            weights = torch.load(weights_file, map_location='cpu')
        else:
            weights = torch.load(weights_file)
        self.ner_model.load_state_dict(weights)
    
    def ner(self, doc):
        """Carry out named entity recognition on doc, finding locations, people, organizations and misc things.

        :param str doc: A string of text, which might have named entities in it.
        :returns dict: Dictionary with keys LOC, PER, ORG and MISC, each of which is a list of found entities.

        :Example:

        >>> from nn.ner import NERModel
        >>> nlp = NERModel(device='cpu')
        >>> nlp.ner('The cat, Kitty, meowed in SOMA')
        {'LOC': ['SOMA'], 'PER': ['Kitty'], 'ORG': [], 'MISC': []}
        """

        tokens = []
        is_heads = []
        tmp_toks = []
        doc = '[CLS] ' + doc + ' [SEP]'
        for word in doc.split():
            tmp = get_tokenizer().tokenize(word.strip()) if word.strip() not in ("[CLS]", "[SEP]") else [word]
            tmp_toks.extend(tmp)
            xx = get_tokenizer().convert_tokens_to_ids(tmp)
            is_head = [1] + [0]*(len(xx) - 1)
            tokens.extend(xx)
            is_heads.extend(is_head)
        tokens = torch.LongTensor([tokens]).to(self.device)
        real_toks = []
        last_tok = ''
        for i, t in enumerate(tmp_toks):
            if is_heads[i]:
                if i != 0:
                    real_toks.append(last_tok)
                last_tok = t
            else:
                last_tok += t.replace('##', '')
        real_toks.append(last_tok)
        _, toktype = self.ner_model(tokens)
        toktype = toktype.cpu().numpy().tolist()
        y_pred = [p for head, p in zip(is_heads, toktype[0]) if head == 1]
        preds = [idx2tag[p] for p in y_pred]
        ents = {
            'LOC': [],
            'PER': [],
            'ORG': [],
            'MISC': [],
        }
        cat = ''
        enttype = ''
        for pred, real_tok in list(zip(preds, real_toks)):
            if pred == 'O':
                if cat and enttype:
                    ents[enttype].append(clean_punctuation(cat))
                cat = ''
                continue
            if pred[0] == 'B':
                cat = real_tok
                enttype = pred.split('-')[-1]
            if pred[0] == 'I':
                cat += ' ' + real_tok
        if cat:
            ents[enttype].append(clean_punctuation(cat))
        return ents