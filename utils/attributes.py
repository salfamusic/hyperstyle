import asyncio
import io
import glob
import os
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw
import dlib
sys.path.append(".")
sys.path.append("..")

from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

def get_face_attributes(filepath, face_client):
    face_attributes = ['age', 'gender', 'headPose', 'smile', 'hair', 'facialHair',  'glasses']

    face_fd = open(filepath, "rb")
    detected_faces = face_client.face.detect_with_stream(face_fd, return_face_attributes=face_attributes)
    if not detected_faces:
        return np.expand_dims(np.array([-1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]),  axis = 1)

    for face in detected_faces:
        gender = 0
        print(str(face.face_attributes.gender))
        if str(face.face_attributes.gender)== 'Gender.male':
            gender = 1

        glasses = 1
        if str(face.face_attributes.glasses)== 'GlassesType.no_glasses':
            glasses = 0

        yaw = face.face_attributes.head_pose.yaw
        pitch = face.face_attributes.head_pose.pitch
        bald = face.face_attributes.hair.bald
        beard =  face.face_attributes.facial_hair.beard
        age = face.face_attributes.age
        expression = face.face_attributes.smile
        return np.expand_dims(np.array([gender,  glasses,  yaw,  pitch,  bald,  beard,  age,  expression]),  axis = 1)