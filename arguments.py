import argparse
import os
cwd = os.path.dirname(os.path.realpath(__file__))  #find directory storing this code

def params():
    parser = argparse.ArgumentParser()

    # Experiment options
    parser.add_argument("--task", default="baseline", type=str,\
                help="baseline is fine-tuning bert for classification;\n\
                      custom is advanced techiques to fine-tune bert;\n\
                      supcon is contrastive learning method",
                      choices=['baseline','custom','supcon'])

    # optional fine-tuning techiques parameters
    parser.add_argument("--reinit_n_layers", default=0, type=int, 
                help="number of layers that are reinitialized. Count from last to first.")
    
    # Others
    parser.add_argument("--input-dir", default=cwd+'/assets', type=str, 
                help="The input training data file (a text file).")
    parser.add_argument("--output-dir", default=cwd+'/results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='bert', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="amazon", type=str,
                help="dataset", choices=['amazon'])
    

    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")
    
    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=16, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--weight-decay", default=0.9, type=float,
                help="Weight decay parameter for scheduler.")
    parser.add_argument("--step-size", default=1, type=int,
                help="Step size parameter for scheduler.")
    parser.add_argument("--learning-rate", default=5e-5, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=10, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.9, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--embed-dim", default=768, type=int, #768 for pre-trained bert
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=10, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=20, type=int,
                help="maximum sequence length to look back")

    args = parser.parse_args()
    return args
