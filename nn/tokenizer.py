from pytorch_pretrained_bert import BertTokenizer

tokenizer = None

def get_tokenizer(bert_model='bert-base-cased', do_lower_case=False):
    global tokenizer
    if tokenizer:
        return tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=do_lower_case)
    return tokenizer

