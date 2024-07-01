import random
import json

def get_response(intent, intents_file):
    with open(intents_file, 'r') as f:
        intents = json.load(f)

    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm not sure how to help with that."
