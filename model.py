import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig, BertForSequenceClassification
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

class ScenarioModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size
    # task1: add necessary class variables as you wish.
    
    # task2: initilize the dropout and classify layers
    self.dropout = nn.Dropout(args.drop_rate)
    self.classify = Classifier(args, target_size)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    
    if args.use_lora:
      print("Setting up LoRA")
      # need to use BertForSequenceClassification when doing PEFT, which is a subclass of BertModel, since
      # there is some internal issue about the forward function getting an implicit labels argument from BertModel
      # that doesn't exist for PEFT, so it crashes
      self.encoder = BertForSequenceClassification.from_pretrained("bert-base-uncased")
      lora_config = LoraConfig(
              r=args.lora_rank,  # Rank of LoRA (experiment with 8, 16, 32)
              lora_alpha=16,  # Scaling factor
              target_modules=["query", "value"],  # we're applying LoRA to these attention layers
              lora_dropout=0.1,
              bias="none",
              task_type=TaskType.FEATURE_EXTRACTION
          )
      self.encoder = get_peft_model(self.encoder, lora_config)
    else:
      self.encoder = BertModel.from_pretrained("bert-base-uncased")
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

  def forward(self, inputs):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
    # task 1: feeding the input to the encoder
    # inputs should contain the input_ids, attention_mask, etc.
    encoder_outputs = self.encoder(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        token_type_ids=inputs.get('token_type_ids', None),  # Handle cases without token type IDs
        output_hidden_states=True
    )
    # task 2: take the last_hidden_state's <CLS> token as output
    # explanation of CLS:
    # The [CLS] (classification) token is a special token that BERT prepends to every input sequence. Unlike regular tokens, it does not correspond to any word in the input. Instead, it is meant to capture a high-level summary representation of the entire sequence.
    # The [CLS] token is special compared to other tokens. It starts as a meaningless embedding and learns to represent whatever the model needs for classification tasks.
    # When training on classification tasks, the gradient updates push [CLS] toward encoding a holistic view of the sequence, while other tokens are optimized for their contextualized word meanings.
    # NOTE: since we are needing to use BertForSequenceClassification when doing PEFT (LoRA), the object properties are different, so we have to check and see (hence the if statement with hasattr)
    last_hidden_state = encoder_outputs.last_hidden_state if hasattr(encoder_outputs, 'last_hidden_state') else encoder_outputs.hidden_states[-1] # shape: (batch_size, sequence_length, hidden_size)
    cls_output = last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size) for <CLS> token

    # task 3: feed the output of the dropout layer to the Classifier
    cls_output = self.dropout(cls_output)  # Apply dropout
    logits = self.classify(cls_output)  # Get predictions from the classifier
    
    return logits  
  
class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size):
    # task1: use initialization for setting different strategies/techniques to better fine-tune the BERT model
    super().__init__(args, tokenizer, target_size)
  
  def get_optimizer_params(self, base_lr, llrd_factor):
    if llrd_factor is None:
      return self.encoder.parameters()

    param_groups = []
    num_layers = len(self.encoder.encoder.layer)

    # Assign decreasing learning rates to lower layers
    for i, layer in enumerate(self.encoder.encoder.layer):
      layer_lr = base_lr * (llrd_factor ** (num_layers - 1 - i))  # Decay LR for lower layers
      param_groups.append({"params": layer.parameters(), "lr": layer_lr})

    # Ensure embeddings and classifier use base LR
    param_groups.append({"params": self.encoder.embeddings.parameters(), "lr": base_lr})
    param_groups.append({"params": self.classify.parameters(), "lr": base_lr})

    return param_groups
  
class SupConModel(ScenarioModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)
    self.args = args
    self.target_size = target_size

    # task1: initialize a linear head layer
    input_dim = args.embed_dim
    self.head = nn.Linear(input_dim, feat_dim)  # Linear head layer -- input_dim x feat_dim, NOT input_dim x target_dim bc we are not classifying!!!
    
  def forward(self, inputs, targets=None):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding
    """
    # task 1: feeding the input to the encoder
    # inputs should contain the input_ids, attention_mask, etc.
    encoder_outputs = self.encoder(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        token_type_ids=inputs.get('token_type_ids', None),  # Handle cases without token type IDs
        output_hidden_states=True
    )
    
    last_hidden_state = encoder_outputs.last_hidden_state # shape: (batch_size, sequence_length, hidden_size)
    cls_output = last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_size) for <CLS> token
    
    # task 3: feed the normalized output of the dropout layer to the linear head layer; return the embedding
    # do NOT dropout and normalize when head becomes classifier!!!
    # logits = self.head(F.normalize(self.dropout(cls_output), dim=1))
    if isinstance(self.head, nn.Linear):
      logits = self.head(F.normalize(self.dropout(cls_output), dim=1))
    else:
      logits = self.head(cls_output)
    
    return logits
  
  def replace_classifier(self, args):
    """Replace the classifier with a new one (trainable)"""
    for param in self.encoder.parameters():
        param.requires_grad = False  # Freeze encoder
    self.head = Classifier(self.args, self.target_size).to(self.encoder.device)  

