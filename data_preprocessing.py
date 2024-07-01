import json
import torch
from transformers import BertTokenizer

def load_intents(file_path):
    with open(file_path, 'r') as f:
        intents = json.load(f)
    return intents

def tokenize_intents(intents, tokenizer, max_length=128):
    input_ids = []
    attention_masks = []
    labels = []

    label_to_id = {intent['tag']: idx for idx, intent in enumerate(intents['intents'])}
    id_to_label = {idx: intent['tag'] for idx, intent in enumerate(intents['intents'])}

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            encoded_dict = tokenizer.encode_plus(
                pattern,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])
            labels.append(label_to_id[intent['tag']])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels, label_to_id, id_to_label
