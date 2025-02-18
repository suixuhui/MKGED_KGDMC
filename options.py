import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-MODEL_DIR', type=str, default='./models')
parser.add_argument('-DATA_DIR', type=str, default='./datasets')
parser.add_argument('-FILE_DIR', type=str, default='./datasets/wn18rr/file')
parser.add_argument('-data_path', type=str, default='./datasets/wn18rr')
parser.add_argument('-img_data_path', type=str, default='./datasets/wnimgs')
parser.add_argument('-dataset', type=str, default='wn18rr')

# model
parser.add_argument("-model", type=str, default='mked')
parser.add_argument('--max_seq_length', type=int, default=128, help='max sequence length for inputs to bert')

# dataset
parser.add_argument("-num_anomalies", type=int, default=4894) # fb15k: 16321
parser.add_argument("-num_triples", type=int, default=97897) # fb15k: 326437
parser.add_argument("-num_entity_anomalies", type=int, default=2047) # fb15k: 727
parser.add_argument("-num_entity", type=int, default=40943) # fb15k: 14541
parser.add_argument("-num_relation", type=int, default=11) # fb15k: 237

# transe
parser.add_argument("-transe_n_epochs", type=int, default=1000)
parser.add_argument("-transe_batch_size", type=int, default=128)
parser.add_argument("-transe_negative_number", type=int, default=3)
parser.add_argument("-transe_lr", type=float, default=1e-4)
parser.add_argument("-transe_threshold", type=int, default=500)

# bert
parser.add_argument("-bert_n_epochs", type=int, default=2)
parser.add_argument("-bert_batch_size", type=int, default=128)
parser.add_argument("-bert_negative_number", type=int, default=5)
parser.add_argument("-bert_lr", type=float, default=3e-5)
parser.add_argument("-bert_threshold", type=int, default=1)

# vit
parser.add_argument("-vit_n_epochs", type=int, default=5)
parser.add_argument("-vit_batch_size", type=int, default=48)
parser.add_argument("-vit_negative_number", type=int, default=2)
parser.add_argument("-vit_lr", type=float, default=3e-5)
parser.add_argument("-vit_threshold", type=int, default=1)

# vae
parser.add_argument("-vae_n_epochs", type=int, default=1000)
parser.add_argument("-vae_batch_size", type=int, default=128)
parser.add_argument("-alpha", type=float, default=0.005)
parser.add_argument("-beta", type=float, default=0.01)
parser.add_argument("-vae_lr", type=float, default=3e-4)
parser.add_argument("-hidden_dim", type=int, default=640)
parser.add_argument("-latent_dim", type=int, default=256)
parser.add_argument("-vae_threshold", type=int, default=300)

# mkged
parser.add_argument("-n_epochs", type=int, default=20)
parser.add_argument("-batch_size", type=int, default=128)
parser.add_argument("-lr", type=float, default=3e-4)
parser.add_argument("-lam", type=float, default=1.0) # fb15k: 0.5
parser.add_argument("-delta", type=float, default=1.5)
parser.add_argument("-num_layers", type=int, default=2)
parser.add_argument("-neighbor_number", type=int, default=59) # fb15k: 119
parser.add_argument("-path_number", type=int, default=10) # fb15k: 20

# other
parser.add_argument('-gama', default=1.0, type=float)
parser.add_argument("-dropout", type=float, default=0.2)
parser.add_argument("-gpu", type=int, default=0, help='-1 if not use gpu, >=0 if use gpu')
parser.add_argument('-num_workers', type=int, default=16, help='num workers for Dataloader')
parser.add_argument('-random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')

parser.add_argument('-bert_dir', type=str, default="bert-base-cased")
parser.add_argument('-vit_dir', type=str, default="vit-base-patch16-224-in21k")

args = parser.parse_args()
command = ' '.join(['python'] + sys.argv)
args.command = command
