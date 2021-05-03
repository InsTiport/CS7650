import json
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertForQuestionAnswering, DistilBertTokenizerFast
from torch.utils.data import DataLoader
import argparse
from transformers import AdamW
import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from dataset import SquadDataset

# setup args
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)

args = arg_parser.parse_args()

'''
hyper-parameter 
'''
DEVICE_ID = args.gpu  # adjust this to use an unoccupied GPU
BATCH_SIZE = args.batch
NUM_EPOCH = args.epoch

'''
control and logging
'''
# control randomness
torch.manual_seed(0)
np.random.seed(0)


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    ids = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                id = qa['id']
                question = qa['question']

                contexts.append(context)
                questions.append(question)
                ids.append(id)

    return contexts, questions, ids


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU

'''
load data
'''
val_contexts, val_questions, val_ids = read_squad('data/dev-v2.0.json')

'''
tokenizers and models
'''
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
model.load_state_dict(torch.load(os.path.join('model_weights', 'BERT_epoch_3.pt'), map_location=device))

model.eval()

'''
tokenize
'''
val_encodings = bert_tokenizer(val_contexts, val_questions, truncation=True, padding=True)
val_encodings['id'] = val_ids

'''
Torch dataset object
'''
val_dataset = SquadDataset(val_encodings, device)

res = dict()
with torch.no_grad():
    for i in tqdm(range(len(val_dataset))):
        id = val_dataset[i]['id']
        input_id = val_dataset[i]['input_ids'].to(device)
        attention_mask = val_dataset[i]['attention_mask'].to(device)
        output = model(
            input_id,
            attention_mask=attention_mask
        )

        start_logits = output.start_logits
        end_logits = output.end_logits
        start_pos = start_logits.argmax(dim=-1)
        end_pos = end_logits.argmax(dim=-1)

        res[id] = val_contexts[start_pos:end_pos]

with open("res.json", "w") as write_file:
    json.dump(res, write_file)
