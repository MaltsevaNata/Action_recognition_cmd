# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json

def create_pose_json(op, opWrapper, image):
    try:
        # Process Image
        # Flags
        parser = argparse.ArgumentParser()
        parser.add_argument("--image_path", default=image)
        # parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg")
        # parser.add_argument("--image_path", default="../../../examples/media/Danya.png", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()
        datum = op.Datum()
        imageToProcess = image  # cv2.imread(args[0].image_path) if using imagepath
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop([datum])

        poseModel = op.PoseModel.BODY_25
        #print(op.getPoseBodyPartMapping(poseModel))

        poseModel = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'MidHip',
                     9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 17: 'REar',
                     18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel', 25: 'Background'}
        merged = {}
        for person in range(len(datum.poseKeypoints)):
            anchor_point_X = datum.poseKeypoints[person][1][0] #Neck
            anchor_point_Y = datum.poseKeypoints[person][1][1]
            persnum = "Person{}".format(person)
            parts = [{'bodypart': poseModel[bodypart], 'X': str(datum.poseKeypoints[person][bodypart][0] - anchor_point_X), 'Y': str(datum.poseKeypoints[person][bodypart][1] - anchor_point_Y),
                    'Confidence': str(datum.poseKeypoints[person][bodypart][2])} for bodypart in range(len(datum.poseKeypoints[person]))]
            '''lefthand = []
            for part in range(int((datum.handKeypoints[0].size)/3)):
                lefthand.append({'X' : str(datum.handKeypoints[0][part][0]), 'Y': str(datum.handKeypoints[0][part][1])})
            leftHands = ([{'LeftHandParts': {'X' : str(datum.handKeypoints[person][0][part][0]), 'Y': str(datum.handKeypoints[person][0][part][1])} for part in range(datum.handKeypoints[0].size)}])
            parts.append(leftHands)'''
           # parts.append({'RightHand': str(datum.handKeypoints[1])})
            merged[persnum] = parts
        res = json.dumps(merged,               sort_keys=False, indent=4, separators=(',', ': '))
        # Display Image
        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        #cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        #cv2.waitKey(0)
        return res
    except Exception as e:
        print(e)
        return -1
