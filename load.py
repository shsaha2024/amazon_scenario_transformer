from transformers import BertTokenizer
import torch

def load_data():
    from datasets import load_dataset
    dataset = load_dataset("mteb/amazon_massive_scenario", "en")
    print(dataset)
    return dataset

def load_tokenizer(args):
    # task1: load bert tokenizer from pretrained "bert-base-uncased", you can also set truncation_side as "left" 
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", torch_dtype=torch.float16, attn_implementation="sdpa")
    return tokenizer

