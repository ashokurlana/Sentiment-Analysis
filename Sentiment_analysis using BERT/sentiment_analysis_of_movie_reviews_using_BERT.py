"""Sentiment Analysis of English movie reviews in NLTK using BERT."""
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from transformers import TextClassificationPipeline
from datasets import Dataset
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=128)

def tokenize_data(data):
    """
    Tokenize data using a pretrained tokenizer.

    Args:
    data: Data in Huggingface format.

    Returns:
    tokenized_data: Tokenized data
    """
    return tokenizer(data["sentence"], truncation=True, padding=True, max_length=128)


def write_lines_to_file(lines, file_path):
    """
    Write lines to a file.

    Args:
    lines: Lines to be written to the file.
    file_path: Enter the output file path.

    Returns: None
    """
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(lines))


def main():
    """
    Pass arguments and call functions here.

    Args: None

    Returns: None
    """
    parser = ArgumentParser(description='This program is about finetuning a sentiment analyzer model.')
    parser.add_argument('--train', dest='tr', help='Enter the training data in CSV format.')
    parser.add_argument('--test', dest='te', help='Enter the test data in CSV format.')
    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--epoch', dest='ep', help='Enter the number of epochs.', type=int)
    parser.add_argument('--output', dest='out', help='Enter the output file path for predictions.')
    args = parser.parse_args()
    # this is an uncased bert-base model
    

    train_dataset = Dataset.from_csv(args.tr, split='train', delimiter=',', header='infer')
    test_dataset = Dataset.from_csv(args.te, split='test', delimiter=',', header='infer')
    num_labels = 2
    print(device)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    # move the model to the same device as the input tensor
    # model.to(device)
    # create the tokenized dataset
    print(train_dataset)
    print(test_dataset)
    train_tokenized_dataset = train_dataset.map(tokenize_data, batched=True)
    test_tokenized_dataset = test_dataset.map(tokenize_data, batched=True)
    training_args = TrainingArguments(
        output_dir="sentiment_analyzer",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=args.ep,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_dataset,
        eval_dataset=test_tokenized_dataset,
        tokenizer=tokenizer
    )
    # train a model with specified arguments
    trainer.train()
    # to predict and return the class/label with the highest score
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    # print the outputs on the evaluation dataset
    model.save_pretrained(args.mod)
    print('Training Done')
    predictions = pipe(test_dataset['sentence'])
    # the below code is to save the most recent model
    actual_labels = []
    for prediction in predictions:
        pred_label = prediction['label']
        pred_index = pred_label.split('_')[1]
        actual_labels.append(pred_index)
    write_lines_to_file(actual_labels, args.out)


if __name__ == '__main__':
    main()