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
