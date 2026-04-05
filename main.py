import os, sys, pdb

from matplotlib.style import use
import numpy as np
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import math
import datetime

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories, plot_accuracy
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import ScenarioModel, SupConModel, CustomModel
from torch import nn
from loss import SupConLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()

    print(f"Running baseline with LoRA set to {args.use_lora}")
    
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, tokenizer, datasets["train"], split='train')
    result_path = f"{args.output_dir}/{args.task}/{'results_LoRA' if args.use_lora else 'results'}.txt" # Create results file if it doesn't exist already to store model performance
    
    # task2: setup model's optimizer_scheduler if you have
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)
    optimizer_scheduler = ExponentialLR(optimizer, gamma= args.weight_decay)
    
    # task3: write a training loop
    train_acc = []
    val_acc = []
    if not os.path.exists(result_path):
      with open(result_path,"w") as file:
         file.write("Model Performance with Fine Tuning at epoch 7\n \n")
    
    for epoch_count in range(args.n_epochs):
        acc = 0
        losses = 0
        model.train()
        startTime = datetime.datetime.now()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels, _ = prepare_inputs(batch)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            optimizer.step()  
            losses += loss.item()
        
        endTime = datetime.datetime.now()
        acc = acc / len(datasets["train"])
        train_acc.append(acc)
        _, acc = run_eval(args, model, datasets, tokenizer, split='validation')
        val_acc.append(acc)
        print('epoch', epoch_count, '| train losses:', losses)

        if (epoch_count + 1) % args.step_size == 0:
            optimizer_scheduler.step()  # update the learning rate schedule
        
        if epoch_count + 1 == 7:
            exec_time = endTime-startTime
            line = f"Experimental Settings: Learning rate = {args.learning_rate}, Drop rate = {args.drop_rate}, Hidden dim = {args.hidden_dim}, Look back = {args.max_len}, Weight Decay = {args.weight_decay}, Step size = {args.step_size}\nTraining time: {exec_time}, Training Accuracy : {acc}, "
            if args.use_lora:
              trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
              total_params = sum(p.numel() for p in model.parameters())
              percent_trainable = 100 * trainable_params / total_params
              line = (
                  line +
                  f"LoRA Rank = {args.lora_rank}\n" +
                  f"Percentage of Trainable Parameters: {percent_trainable:.2f}%\n" 
              ) 
            f = open(result_path,"a")
            f.write(line)
    plot_accuracy(train_acc, val_acc, fname = f"baseline_train_LoRA_{args.lora_rank}" if args.use_lora else "baseline_train")
  
def custom_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    best_model = None
    best_val_loss = math.inf

    # task1: setup train dataloader
    train_dataloader = get_dataloader(args, tokenizer, datasets["train"], split='train')
    result_path = f"{args.output_dir}/{args.task}/results.txt" 
    
    # task2: setup model's optimizer_scheduler if you have
    llrd_factor = 0.9  # layer-wise learning rate decay factor
    base_lr = args.learning_rate
    optimizer = optim.AdamW(model.get_optimizer_params(base_lr, llrd_factor), lr=base_lr)
    optimizer_scheduler = ExponentialLR(optimizer, gamma= args.weight_decay)
  
    # task3: write a training loop
    train_acc = []
    val_acc = []
    if not os.path.exists(result_path):
      with open(result_path,"w") as file:
         file.write("Model Performance with Custom Fine Tuning\n \n")
    
    for epoch_count in range(args.n_epochs):
        acc = 0
        losses = 0
        model.train()
        freq_val_steps = 16
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels, _ = prepare_inputs(batch)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
            optimizer.step()  
            losses += loss.item()
            if args.use_freq_val and step % freq_val_steps == 0:
              val_loss, _ = run_eval(args, model, datasets, tokenizer, split='validation')
              if val_loss < best_val_loss:
                best_model = model
                val_loss = best_val_loss
        
        acc = acc / len(datasets["train"])
        train_acc.append(acc)
        val_loss, acc = run_eval(args, model, datasets, tokenizer, split='validation')
        val_acc.append(acc)
        if not args.use_freq_val:
          if val_loss < best_val_loss:
            best_model = model
            val_loss = best_val_loss
        print('epoch', epoch_count, '| train losses:', losses)

        if (epoch_count + 1) % args.step_size == 0:
          optimizer_scheduler.step()  # update the learning rate schedule
          
        if epoch_count + 1 == 7:
            line = f"Experimental Settings: Base Learning rate = {args.learning_rate}, Drop rate = {args.drop_rate}, Hidden dim = {args.hidden_dim}, Look back = {args.max_len}, Weight Decay = {args.weight_decay}, Step size = {args.step_size}, "
            if args.use_llrd:
              line += f"LLRD factor = {llrd_factor}, "
            if args.use_freq_val:
              line +=  f"Frequent validation steps = {freq_val_steps}, "
            line += "\n"
            f = open(result_path,"a")
            f.write(line)

    fname = "custom_train"
    if args.use_llrd:
      fname += "_llrd"
    if args.use_freq_val:
      fname += "_freq_val"
    plot_accuracy(train_acc, val_acc, fname = fname)

    return best_model

def run_eval(args, model, datasets, tokenizer, split='validation', should_print=False):#print paramter if you would like test accuracy written into the results file
    criterion = nn.CrossEntropyLoss()
    model.eval()
    dataloader = get_dataloader(args, tokenizer, datasets[split], split)
    losses = 0
    acc = 0
    for step, batch in enumerate(dataloader):
        inputs, labels, _ = prepare_inputs(batch)
        logits = model(inputs)
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
        loss = criterion(logits, labels)
        losses += loss.item()
    print(f'{split} acc:', acc/len(datasets[split]), f"| {split} loss:", losses , f'|dataset split {split} size:', len(datasets[split]))
    if split == "test" and should_print:
        result_path = f"{args.output_dir}/{args.task}"
        if args.use_lora:
          result_path += "/results_LoRA.txt"
        elif args.use_sim:
          result_path += "/results_simCLR.txt"
        elif args.use_supcon:
          result_path += "/results_supcon.txt"
        else: 
          result_path += "/results.txt"
        line = f"Test loss: {losses}, Test accuracy: {acc/len(datasets[split])}\n \n"
        f = open(result_path,"a")
        f.write(line)
        
    return losses/len(datasets[split]), acc/len(datasets[split])

def supcon_train(args, model, datasets, tokenizer):
    criterion = SupConLoss()
    best_model = None
    best_val_loss = math.inf

    # task1: load training split of the dataset
    train_dataloader = get_dataloader(args, tokenizer, datasets["train"], split='train')
    result_path = f"{args.output_dir}/{args.task}/{'results_simCLR' if args.use_sim else 'results_supcon'}.txt" # Create results file if it doesn't exist already to store model performance
    
    # task2: setup optimizer_scheduler in your model
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)
    optimizer_scheduler = ExponentialLR(optimizer, gamma= args.weight_decay)
  
    # task3: write a training loop for SupConLoss function 
    train_accs = []
    val_accs = []
    if not os.path.exists(result_path):
      with open(result_path,"w") as file:
         file.write("Model Performance with SupCon/SimClear \n \n")

    for epoch_count in range(args.n_epochs):
        if epoch_count == 5:
          model.replace_classifier(args)
          if args.use_sim:
            args.learning_rate = 1e-3
            optimizer_scheduler = None
          optimizer = optim.AdamW(model.head.parameters(), lr=args.learning_rate)
          criterion = nn.CrossEntropyLoss()
          args.use_supcon = False # bc training Classifier head now
          print("Freeze and add classifier")
          
        acc = 0
        losses = 0
        model.train()
        
        for step, batch in progress_bar(enumerate(train_dataloader), total=len(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels, _ = prepare_inputs(batch)
            # Generate two augmented views by running two forward passes.
            logits = model(inputs)
            tem = (logits.argmax(1) == labels).float().sum()
            acc += tem.item()
           
            if args.use_supcon:
              logits2 = model(inputs)
              # Stack the two views to get a tensor of shape [batch_size, 2, feat_dim]
              features = torch.stack([logits, logits2], dim=1)
              if args.use_sim:
                loss = criterion(features)
              else:
                loss = criterion(features, labels)
              tem1 = (logits2.argmax(1) == labels).float().sum()
              acc += tem1.item()
            else:
              loss = criterion(logits, labels) 
            
            loss.backward()
            optimizer.step()
            losses += loss.item()

        val_loss, val_acc = run_eval(args, model, datasets, tokenizer, split='validation')
        val_accs.append(val_acc)
        if val_loss < best_val_loss:
          best_model = model
          val_loss = best_val_loss
        
        acc = acc / (2*len(datasets["train"]))
        train_accs.append(acc)
        
        print('epoch', epoch_count, '| train losses:', losses)

        if (epoch_count + 1) % args.step_size == 0 and (not args.use_supcon and not args.use_simclr):
            optimizer_scheduler.step()  # update the learning rate schedule, only for supcon, we removed scheduler for simclr

    plot_accuracy(train_accs, val_accs, fname="simCLR_train" if args.use_sim else "supcon_train")
    return best_model
        
if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)
  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))

  args.use_lora = False
  args.use_llrd = False # Layer-wise Learning Rate Decay
  args.use_freq_val = False # Frequent validation
  args.use_sim = False
  args.use_supcon = False
  if args.task == 'baseline':
    args.use_lora = True
    args.lora_rank = 32 # try 8, 16, 32
    args.n_epochs = 20
    model = ScenarioModel(args, tokenizer, target_size=18).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test', should_print=True)
  elif args.task == 'custom': # you can have multiple custom task for different techniques
    args.use_llrd = True
    args.use_freq_val = True
    model = CustomModel(args, tokenizer, target_size=18).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    best_model = custom_train(args, model, datasets, tokenizer)
    run_eval(args, best_model, datasets, tokenizer, split='test', should_print=True)
  elif args.task == 'supcon':
    args.use_sim = True
    args.use_supcon = True # this should always be set to true here
    model = SupConModel(args, tokenizer, target_size=18).to(device)
    args.n_epochs = 20 # change back to 10 for supcon
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    best_model = supcon_train(args, model, datasets, tokenizer)
    args.use_supcon = True # this gets set to false after the head is replace but need to set back to true for eval for filenaming
    run_eval(args, best_model, datasets, tokenizer, split='test', should_print=True)
