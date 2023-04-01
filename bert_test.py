import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# Load the saved model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model_path = "finetuned_BERT_epoch_1.model"
model.load_state_dict(torch.load(model_path))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Example sentences
sentences = ["I am a good boy", "I am a bad boy"]

for sentence in sentences:
    # Tokenize the sentence
    encoded_sentence = tokenizer.encode(sentence, add_special_tokens=True)

    # Convert the token IDs into a tensor
    input_ids = torch.tensor([encoded_sentence])

    # Obtain the predicted class probabilities
    outputs = model(input_ids)
    logits = outputs[0]
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    # Find the index of the maximum probability
    pred_class = np.argmax(probs)

    # Print the predicted class
    print(pred_class)
