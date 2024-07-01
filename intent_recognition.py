import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from src.model_definition import ChatbotModel

def predict_intent(model, tokenizer, sentence, id_to_label, max_length=128):
    encoded_dict = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probabilities, dim=1)

    return id_to_label[top_class.item()], top_prob.item()
