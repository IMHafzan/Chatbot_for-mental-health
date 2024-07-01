import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, AdamW
from src.data_preprocessing import load_intents, tokenize_intents
from src.model_definition import ChatbotModel

def main():
    intents = load_intents(os.path.join(os.path.dirname(__file__), '../data/intents.json'))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_masks, labels, label_to_id, id_to_label = tokenize_intents(intents, tokenizer)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    model = ChatbotModel(num_labels=len(label_to_id))
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(4):
        model.train()
        for batch in train_dataloader:
            b_input_ids, b_attention_masks, b_labels = batch
            optimizer.zero_grad()
            outputs = model(b_input_ids, b_attention_masks)
            loss = torch.nn.CrossEntropyLoss()(outputs, b_labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), '../models/chatbot_model.pth'))
    torch.save(label_to_id, os.path.join(os.path.dirname(__file__), '../models/label_to_id.pth'))
    torch.save(id_to_label, os.path.join(os.path.dirname(__file__), '../models/id_to_label.pth'))

if __name__ == "__main__":
    main()
