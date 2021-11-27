import cv2
from django.core.files.storage import default_storage
from django.shortcuts import render
from rest_framework.views import APIView
import numpy as np
from rest_framework.response import Response
from rest_framework import status
from api.recognition import plant_leaf_disease_recognition
from api.models import Disease
from api.serializers import *


class PredictAPIView(APIView):
    """ PredictAPIView : APIView for get predictions """

    def get_object(self, **kwargs):
        try:
            disease = kwargs['disease']
            # print('\n\n\n\ Passed Disease:', disease)
        except KeyError:
            return Response(status=status.HTTP_404_NOT_FOUND)

        try:
            disease = Disease.objects.get(name=disease)
        except Disease.DoesNotExist:
            context = {
                'error': 'Records not found for the disease.'
            }
            return Response(context, status=status.HTTP_404_NOT_FOUND)
        # print('\n\n\n\n Queried disease:', disease)
        return disease

    def get_disease_details(self, **kwargs):
        try:
            disease = kwargs['disease']
            # print('\n\n\n\n Passed to method:', disease)
        except KeyError:
            return Response(status=status.HTTP_404_NOT_FOUND)

        disease = self.get_object(disease=disease)

        serializer = DiseaseSerializer(disease)

        # print('\n\n\n\n Serialized', type(serializer.data))

        return serializer.data

    def post(self, request, *args, **kwargs):
        file = request.data.get('image')

        # save file into storage
        file_name = default_storage.save('image.jfif', file)
        file_path = default_storage.path(file_name)

        # print(type(file_path), file_path)

        model = plant_leaf_disease_recognition()
        prediction = model.predict(file_path).replace(' ', '_').lower()

        sts = status.HTTP_200_OK

        if prediction != 'helthy_leaf':
            try:
                detail = self.get_disease_details(disease=prediction)
                sts = status.HTTP_200_OK
            except AttributeError:
                detail = "Disease records not found!"
                sts = status.HTTP_404_NOT_FOUND
        else:
            detail = 'N/A'
            sts = status.HTTP_200_OK

        context = {
            'prediction': prediction,
            'detail': detail
        }

        # remove file from storage
        default_storage.delete(file_path)

        return Response(context, status=sts)