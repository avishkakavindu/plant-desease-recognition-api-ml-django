import cv2
from django.core.files.storage import default_storage
from django.shortcuts import render
from rest_framework.views import APIView
import numpy as np
from rest_framework.response import Response
from rest_framework import status


class PredictAPIView(APIView):
    pass