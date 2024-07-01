import os
import torch
from transformers import BertTokenizer
from src.model_definition import ChatbotModel
from src.intent_recognition import predict_intent
from src.response_generation import get_response


def main():
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load the label mappings
    label_to_id = torch.load(os.path.join(os.path.dirname(__file__), '../models/label_to_id.pth'))
    id_to_label = torch.load(os.path.join(os.path.dirname(__file__), '../models/id_to_label.pth'))

    model = ChatbotModel(num_labels=len(label_to_id))
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), '../models/chatbot_model.pth')))
    model.eval()

    print("Chatbot is ready to chat! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        intent, prob = predict_intent(model, tokenizer, user_input, id_to_label)
        response = get_response(intent, os.path.join(os.path.dirname(__file__), '../data/intents.json'))
        print(f"Bot: {response} (Confidence: {prob:.2f})")


if __name__ == "__main__":
    main()
