import json
from pathlib import Path
from tqdm import tqdm
from transformers import BartTokenizer, BartForQuestionAnswering
import argparse
import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# setup args
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)

arg_parser.add_argument(
    '-e', '--epoch',
    type=int,
    default=5,
    help=f'Specify number of training epochs'
)

args = arg_parser.parse_args()

'''
hyper-parameter 
'''
DEVICE_ID = args.gpu  # adjust this to use an unoccupied GPU
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
bert_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForQuestionAnswering.from_pretrained('facebook/bart-base').to(device)
model.load_state_dict(torch.load(os.path.join('model_weights', f'BART_epoch_{NUM_EPOCH}.pt'), map_location=device))

model.eval()


res = dict()
with torch.no_grad():
    for i, (context, question, id) in tqdm(enumerate(zip(val_contexts, val_questions, val_ids))):
        encoding = bert_tokenizer(context, question, return_tensors='pt')
        inputs = encoding['input_ids'].to(device)

        try:
            output = model(inputs)

            start_logits = output.start_logits
            end_logits = output.end_logits
            start_pos = start_logits.argmax(dim=-1)
            end_pos = end_logits.argmax(dim=-1)

            res[id] = context[start_pos:end_pos]
        except RuntimeError:
            res[id] = ''

with open('BART_res.json', 'w') as write_file:
    json.dump(res, write_file)
