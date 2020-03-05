import logging
import urllib

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertConfig

from zincbase.nn.tokenizer import get_tokenizer
from zincbase.utils.file_utils import get_cache_dir, check_file_exists
from zincbase.utils.string_utils import clean_punctuation
from zincbase.utils.misc_utils import chunk

TOKEN_TYPES = ('<PAD>', 'O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC')
tag2idx = {tag: idx for idx, tag in enumerate(TOKEN_TYPES)}
idx2tag = {idx: tag for idx, tag in enumerate(TOKEN_TYPES)}

class BertNER(nn.Module):
    def __init__(self, vocab_size=None, device='cpu', training=False):
        super().__init__()
        bert_vocab_size = 30522
        config = BertConfig(bert_vocab_size, max_position_embeddings=512)
        self.bert = BertModel(config).from_pretrained('bert-base-cased').to(device)
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

        ents = self._ner_inner(doc)
        def got_none(x):
            if (len(x['PER']) + len(x['LOC']) + len(x['ORG']) + len(x['MISC'])) == 0:
                return True
            all_length_one_or_none = True
            for key in x:
                if len(x[key]):
                    for ent in x[key]:
                        if ent and len(ent.split()) != 1:
                            all_length_one_or_none = False
            return all_length_one_or_none

        for key in ents.keys():
            for ent in ents[key]:
                if len(ent.split()) > 1:
                    doc = doc.replace(ent, ' [SEP] ')
        newents = self._ner_inner(doc)
        while not got_none(newents):
            for key in newents:
                ents[key].extend(newents[key])
            for key in ents.keys():
                for ent in ents[key]:
                    if len(ent.split()) > 1:
                        doc = doc.replace(ent, ' [SEP] ')
            newents = self._ner_inner(doc)
        for key in ents:
            ents[key] = list(set(ents[key]))
        return ents

    def _ner_inner(self, doc):
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

        toktype = []
        if len(tokens) > 512:
            logging.warning("Breaking up the document which has 512 tokens. This will reduce accuracy - use shorter docs if possible.")
        for toks in chunk(tokens, 512):
            toks = torch.LongTensor([toks]).to(self.device)
            _, preds = self.ner_model(toks)
            preds = preds.cpu().numpy().tolist()
            toktype.extend(preds[0])

        y_pred = [p for head, p in zip(is_heads, toktype) if head == 1]
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
        for key in ents:
            ents[key] = list(set(ents[key]))
        return ents
