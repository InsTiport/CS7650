import os
import json
import torch
from pathlib import Path
from transformers import BartTokenizer, BartForQuestionAnswering, DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import AdamW
from dataset import SquadDataset


def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers


def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters


def add_token_positions(encodings, bert_encodings, answers):
    start_pos = []
    end_pos = []
    for i in range(len(answers)):
        start_pos.append(bert_encodings.char_to_token(i, answers[i]['answer_start']))
        end_pos.append(bert_encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_pos[-1] is None:
            start_pos[-1] = bart_tokenizer.model_max_length
        if end_pos[-1] is None:
            end_pos[-1] = bert_encodings.char_to_token(i, answers[i]['answer_end'] + 1)

    encodings.update({'start_positions': start_pos, 'end_positions': end_pos})


'''
load data
'''
train_contexts, train_questions, train_answers = read_squad('data/train-v2.0.json')
val_contexts, val_questions, val_answers = read_squad('data/dev-v2.0.json')

'''
generate answer end indices
'''
add_end_idx(train_answers, train_contexts)
add_end_idx(val_answers, val_contexts)

'''
tokenizers and models
'''
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = BartForQuestionAnswering.from_pretrained('facebook/bart-base')

'''
tokenize
'''
train_encodings = bart_tokenizer(train_contexts, train_questions, truncation=True, padding=True)
val_encodings = bart_tokenizer(val_contexts, val_questions, truncation=True, padding=True)
bert_train_encodings = bert_tokenizer(train_contexts, train_questions, truncation=True, padding=True)
bert_val_encodings = bert_tokenizer(val_contexts, val_questions, truncation=True, padding=True)


'''
last step preparing model inputs
'''
add_token_positions(train_encodings, bert_train_encodings, train_answers)
add_token_positions(val_encodings, bert_val_encodings, val_answers)


'''
Torch dataset object
'''
train_dataset = SquadDataset(train_encodings)
val_dataset = SquadDataset(val_encodings)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )
        loss = outputs[0]
        loss.backward()
        optim.step()

    SAVE_PATH = os.path.join('model_weights', f'BART_epoch_{epoch+1}.pt')
    # save model after training for one epoch
    torch.save(model.state_dict(), SAVE_PATH)
