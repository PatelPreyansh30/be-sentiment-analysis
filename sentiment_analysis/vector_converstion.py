import numpy as np


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
