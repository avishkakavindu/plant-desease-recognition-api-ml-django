from rest_framework import serializers
from rest_framework.fields import SerializerMethodField

from api.models import *


class SolutionSerializer(serializers.ModelSerializer):
    """ Serializer for Solution model """

    class Meta:
        model = Solution
        fields = ['solution']
        

class SymptomSerializer(serializers.ModelSerializer):
    """ Serializer for Symptom model """
    
    class Meta:
        model = Symptom
        fields = ['symptom']


class DiseaseSerializer(serializers.ModelSerializer):
    """ Serializer for Disease model """

    symptoms = serializers.SerializerMethodField(read_only=True)
    solutions = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Disease
        fields = ['name', 'pathogen', 'symptoms', 'solutions']

    def get_symptoms(self, obj):
        symptoms = Symptom.objects.filter(disease=obj)
        serializer = SymptomSerializer(symptoms, many=True)
        return serializer.data
    
    def get_solutions(self, obj):
        solutions = Solution.objects.filter(disease=obj)
        serializer = SolutionSerializer(solutions, many=True)
        return serializer.data
