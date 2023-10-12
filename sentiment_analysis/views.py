import pandas as pd
import json
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from rest_framework import views, response, status
from . import serializer, vector_converstion

# Set your model here
model_path = "./model/First_model"
tokenizer_path = "./model/First_model_tokenizer"

print("MODEL AND TOKENIZER GLOBALLY IMPORT STARTING")
model = TFRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
print("MODEL AND TOKENIZER GLOBALLY IMPORT ENDING")


class SentimentAnalysisView(views.APIView):
    def post(self, request, format=None):
        serializer_data = serializer.SentimentAnalysisSerializer(
            data=request.data)
        if serializer_data.is_valid():
            sentence = serializer_data.data.get('sentence')
            result = vector_converstion.tokenizer_analysis(
                sentence, model, tokenizer)
            return response.Response({"status": "success", "result": result}, status.HTTP_201_CREATED)
        return response.Response({"status": "error"}, status.HTTP_400_BAD_REQUEST)


class BulkSentimentAnalysisView(views.APIView):
    def post(self, request, *args, **kwargs):
        serializer_data = serializer.BulkSentimentAnalysisSerializer(
            data=request.data)
        serializer_data.is_valid(raise_exception=True)
        file = serializer_data.validated_data.get('file')

        if file.name.endswith('.txt'):
            with file.open('r') as txt_file:
                reviews = txt_file.read().splitlines()
                result = vector_converstion.analyze_bulk_data(
                    reviews, model, tokenizer)
        elif file.name.endswith('.csv'):
            df = pd.read_csv(file)
            if 'review_text' not in df.columns:
                return response.Response({"status": "error", "result": "CSV file must contain a 'review_text' column"},
                                         status.HTTP_400_BAD_REQUEST)
            reviews = df['review_text'].tolist()
            result = vector_converstion.analyze_bulk_data(
                reviews, model, tokenizer)
        elif file.name.endswith('.json'):
            with file.open('r') as json_file:
                data = json.load(json_file)
            if 'reviews' not in data:
                return response.Response({"status": "error", "result": "JSON file must contain a 'reviews' key with a array of review texts"},
                                         status.HTTP_400_BAD_REQUEST)
            reviews = data['reviews']
            result = vector_converstion.analyze_bulk_data(
                reviews, model, tokenizer)
        else:
            return response.Response({"status": "error", "result": "Unsupported file format. Supported formats: .csv, .json, .txt"},
                                     status.HTTP_400_BAD_REQUEST)

        return response.Response({"status": "success", "result": result},
                                 status.HTTP_201_CREATED)
