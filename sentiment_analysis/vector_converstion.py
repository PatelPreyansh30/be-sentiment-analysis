import numpy as np
import tensorflow as tf


def tokenizer_analysis(sentence, model, tokenizer):
    # Convert to string
    data = str(sentence)

    # Convert string to data matrix
    data_encodings = tokenizer(
        data, padding=True, truncation=True, return_tensors="tf", max_length=128)
    input_ids = data_encodings["input_ids"]
    attention_mask = data_encodings["attention_mask"]

    # Predict the sentence
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
    return {"type": output, "scaled_probability": scaled_probabilities[predicted_label]}


def analyze_bulk_data(reviews, model, tokenizer):
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
        "review_number": [
            {
                "label": "Positive",
                "count": sentiment_counts['Positive']
            },
            {
                "label": "Neutral",
                "count": sentiment_counts['Neutral']
            },
            {
                "label": "Negative",
                "count": sentiment_counts['Negative']
            },
        ],
        "review_percentage": [
            {
                "label": "Positive",
                "count": "{:.2f}".format(percentages["Positive"])
            },
            {
                "label": "Neutral",
                "count": "{:.2f}".format(percentages["Neutral"])
            },
            {
                "label": "Negative",
                "count": "{:.2f}".format(percentages["Negative"])
            },
        ],
        "review_average_probability": [
            {
                "label": "Positive",
                "count": "{:.2f}".format(average_probabilities["Positive"][2]*100)
            },
            {
                "label": "Neutral",
                "count": "{:.2f}".format(average_probabilities["Neutral"][1]*100)
            },
            {
                "label": "Negative",
                "count": "{:.2f}".format(average_probabilities["Negative"][0]*100)
            },
        ]
    }

    return analysis_results
