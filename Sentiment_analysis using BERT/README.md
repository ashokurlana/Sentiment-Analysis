## Step-1 Get the train and test files from original data
```
python3 get_train_test.py
```
## Step-2 Fine-tune the model
```
python3 sentiment_analysis_of_movie_reviews_using_BERT.py --train train.csv --test test.csv --model "bert-base-uncased" --epoch 1 --output predictions.txt
```
## Step-3 Get the Predictions on Test set
```
python3 predict_sentiments_using_huggingface_saved_model.py --model bert-base-uncased/ --test test.csv --output predictions.txt
```
## Fine-tuned BERT model path
https://drive.google.com/file/d/19PYGGyGxw-tmdvTbUqEA1H3wPQqjXIrl/view?usp=sharing
