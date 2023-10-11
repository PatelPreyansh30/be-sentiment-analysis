from rest_framework import serializers


class SentimentAnalysisSerializer(serializers.Serializer):
    sentence = serializers.CharField()

class BulkSentimentAnalysisSerializer(serializers.Serializer):
    file = serializers.FileField()
