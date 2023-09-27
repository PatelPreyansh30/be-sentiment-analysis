import numpy as np
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from rest_framework import views, response, status
from . import serializer

# Set your model here
model_path = "./model/First_model"
tokenizer_path = "./model/First_model_tokenizer"

print("MODEL AND TOKENIZER GLOBALLY IMPORT STARTING")
model = TFRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
print("MODEL AND TOKENIZER GLOBALLY IMPORT ENDING")


def tokenizer_analysis(data):
    # Convert to string
    data = str(data)

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


class SentimentAnalysisView(views.APIView):
    def post(self, request, format=None):
        serializer_data = serializer.SentimentAnalysisSerializer(
            data=request.data)
        if serializer_data.is_valid():
            result = tokenizer_analysis(serializer_data.data.get('sentence'))
            return response.Response({"status": "success", "result": result}, status.HTTP_201_CREATED)
        return response.Response({"status": "error"}, status.HTTP_400_BAD_REQUEST)
