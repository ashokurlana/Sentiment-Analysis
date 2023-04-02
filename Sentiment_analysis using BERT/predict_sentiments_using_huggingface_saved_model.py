"""Predict sentiments using Hugging Face saved models."""
from argparse import ArgumentParser
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TextClassificationPipeline
from tqdm import tqdm

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


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
    parser = ArgumentParser(description='This program is about predicting the sentiments using a saved Hugging Face model.')
    parser.add_argument('--test', dest='te', help='Enter the test data in CSV format.')
    parser.add_argument('--model', dest='mod', help='Enter the model directory.')
    parser.add_argument('--output', dest='out', help='Enter the output file path for predictions.')
    args = parser.parse_args()
    # model path is a directory in Huggingface
    loaded_model = AutoModelForSequenceClassification.from_pretrained(args.mod)
    test_dataset = Dataset.from_csv(args.te, split='test', delimiter=',', header='infer')
    # 2 ways to predict: 1 with pipeline, the other being passing inputs to the model
    pipe = TextClassificationPipeline(model=loaded_model, tokenizer=tokenizer)
    # print the outputs on the test dataset
    predictions = pipe(test_dataset['sentence'])
    actual_labels = []
    for prediction in tqdm(predictions):
        pred_label = prediction['label']
        pred_index = pred_label.split('_')[1]
        actual_labels.append(pred_index)
    write_lines_to_file(actual_labels, args.out)


if __name__ == '__main__':
    main()
