import numpy as np
# import io, csv, pandas as pd
import tensorflow as tf
# from joblib import load
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from rest_framework import views, response, status
from . import serializer, vector_converstion

# Set your model here
model_path = "./model/First_model"
tokenizer_path = "./model/First_model_tokenizer"

print("MODEL AND TOKENIZER GLOBALLY IMPORT STARTING")
# sahil = load("./model/sentiment_analysis.joblib")
model = TFRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
print("MODEL AND TOKENIZER GLOBALLY IMPORT ENDING")


def analyze_bulk_data(reviews):
    batch_size = 256  # You can adjust this value based on your available GPU memory

    corpus = [str(review) for review in reviews]

    # Initialize lists to store results
    sentiment_counts = {
        "Negative": 0,
        "Neutral": 0,
        "Positive": 0
    }

    sentiment_probabilities = {
        "Negative": [],
        "Neutral": [],
        "Positive": []
    }

    # Process data in batches
    for i in range(0, len(corpus), batch_size):
        batch_corpus = corpus[i:i+batch_size]

        # Tokenize and encode reviews in batches
        data_encodings = tokenizer(batch_corpus, padding=True, truncation=True,
                                   return_tensors="tf", max_length=128, return_attention_mask=True)

        # Make predictions in batches
        predictions = model(
            {'input_ids': data_encodings['input_ids'], 'attention_mask': data_encodings['attention_mask']}).logits

        # Extract sentiment labels for each review
        predicted_labels = tf.argmax(predictions, axis=1).numpy()

        for label, probabilities in zip(predicted_labels, tf.nn.softmax(predictions).numpy()):
            sentiment = "Negative" if label == 0 else "Neutral" if label == 1 else "Positive"
            sentiment_counts[sentiment] += 1
            sentiment_probabilities[sentiment].append(probabilities)

    total_reviews = len(corpus)
    percentages = {k: v / total_reviews *
                   100 for k, v in sentiment_counts.items()}

    # Calculate and print average probabilities
    softmax_probs = tf.nn.softmax(predictions, axis=-1).numpy()
    average_probabilities = {
        "Negative": np.mean(softmax_probs[predicted_labels == 0], axis=0),
        "Neutral": np.mean(softmax_probs[predicted_labels == 1], axis=0),
        "Positive": np.mean(softmax_probs[predicted_labels == 2], axis=0)
    }

    analysis_results = {
        'Number of positive Reviews': sentiment_counts['Positive'],
        'Number of neutral Reviews': sentiment_counts['Neutral'],
        'Number of negative Reviews': sentiment_counts['Negative'],
        'Percentage of Positive reviews in data': "{:.2f}".format(percentages["Positive"]),
        'Percentage of neutral reviews in data': "{:.2f}".format(percentages["Neutral"]),
        'Percentage of negative reviews in data': "{:.2f}".format(percentages["Negative"]),
        'Average probabilities of positive reviews': "{:.4f}".format(average_probabilities["Positive"][0]),
        'Average probabilities of neutral reviews': "{:.4f}".format(average_probabilities["Neutral"][0]),
        'Average probabilities of negative reviews': "{:.4f}".format(average_probabilities["Negative"][0]),
    }

    return analysis_results


def analysis(data):
    data = str(data)
    data_encodings = tokenizer(
        data, padding=True, truncation=True, return_tensors="tf", max_length=128)
    input_ids = data_encodings["input_ids"]
    attention_mask = data_encodings["attention_mask"]
    predictions = model.predict(
        {"input_ids": input_ids, "attention_mask": attention_mask})
    predicted_label = np.argmax(predictions[0])
    logits = predictions.logits[0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    scaled_probabilities = probabilities * 100

    if predicted_label == 0:
        output = "Negative"
    elif predicted_label == 1:
        output = "Neutral"
    else:
        output = "Positive"
    return {"type": output, "prediction": predictions, "scaled_probability": scaled_probabilities[predicted_label]}


class SentimentAnalysisView(views.APIView):
    def post(self, request, format=None):
        serializer_data = serializer.SentimentAnalysisSerializer(
            data=request.data)
        if serializer_data.is_valid():
            sentence = serializer_data.data.get('sentence')
            result = vector_converstion.tokenizer_analysis(sentence, model, tokenizer)
            # vector = vector_converstion.naive_byse_classification(
            #     serializer_data.data.get('sentence'))
            # print(vector)
            # result = sahil.predict(vector)
            return response.Response({"status": "success", "result": result}, status.HTTP_201_CREATED)
        return response.Response({"status": "error"}, status.HTTP_400_BAD_REQUEST)


class BulkSentimentAnalysisView(views.APIView):

    def post(self, request, *args, **kwargs):
        serializer_data = serializer.BulkSentimentAnalysisSerializer(
            data=request.data)
        serializer_data.is_valid(raise_exception=True)
        file = serializer_data.validated_data['file']
        with file.open('r') as txt_file:
            reviews = txt_file.read().splitlines()
            result = analyze_bulk_data(reviews)

        return response.Response({"status": "success", "result": result},
                                 status.HTTP_201_CREATED)
