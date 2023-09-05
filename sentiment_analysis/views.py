from django.core.cache import cache
import numpy as np
from rest_framework import views, response, status
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from . import serializer
# import tensorflow as tf



def get_model():
    model_path = "./model/First_model"
    # load_options = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    # model = tf.saved_model.load(model_path, options=load_options)
    model = cache.get("model")
    if model is None:
        model = TFRobertaForSequenceClassification.from_pretrained(model_path)
        cache.set("model", model)
        print("Model running")
    return model


def get_tokenizer():
    tokenizer = cache.get("tokenizer")
    if tokenizer is None:
        tokenizer_path = "./model/First_model_tokenizer"
        tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
        cache.set("tokenizer", tokenizer)
        print("Tokenizer running")
    return tokenizer


def analysis(data):
    model = get_model()
    tokenizer = get_tokenizer()

    data = str(data)
    data_encodings = tokenizer(
        data, padding=True, truncation=True, return_tensors="tf", max_length=128)
    input_ids = data_encodings["input_ids"]
    attention_mask = data_encodings["attention_mask"]
    predictions = model.predict(
        {"input_ids": input_ids, "attention_mask": attention_mask})
    predicted_label = np.argmax(predictions[0])

    if predicted_label == 0:
        output = "Negative"
    elif predicted_label == 1:
        output = "Neutral"
    else:
        output = "Positive"
    return output


class SentimentAnalysisView(views.APIView):
    def post(self, request, format=None):
        serializer_data = serializer.SentimentAnalysisSerializer(
            data=request.data)
        if serializer_data.is_valid():
            result = analysis(serializer_data.data.get('sentence'))
            return response.Response({"message": "Success", "result": result}, status.HTTP_201_CREATED)
        return response.Response({"message": "Error"}, status.HTTP_400_BAD_REQUEST)
