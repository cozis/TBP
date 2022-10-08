import torch
import numpy as np
import pandas as pd
import os
import cv2
import time
from tqdm import tqdm
from sys import argv
import sys
from retinaface.detect_retina import get_detector, detect_video_torch
from retinaface.boxes_utils import convert_to_square_margin, roi_linear

def main_detect_face(filevideo, fileboxes, num, gpu = 0):

    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu"  , type=int, default=0, help="gpu id")
    parser.add_argument("-i", "--input", type=str, help="input video")
    parser.add_argument("-b", "--boxes", type=str, help="input file with box")
    parser.add_argument("-n", "--num"  , type=int, default=0, help="number of frames to process")
    args = parser.parse_args()
    '''

    target_size = 1280; stack = 8
    device = torch.device('cuda:%d'%gpu if torch.cuda.is_available() and gpu >= 0 else 'cpu')
    length = 0
    skip   = 0

    print('Running on device: {}'.format(device))
    print('input : {}'.format(filevideo))
    print('output: {}'.format(fileboxes))

    # Define retinaface network
    retinaface = get_detector(network='resnet50', device=device, trained_model=None).eval()
    def detect_faces(frames):
        resize_factor = min(target_size / max(frames.shape[-1],frames.shape[-2]) , 1.0)

        return detect_video_torch(retinaface, frames, resize_factor=resize_factor,
                                  score_threshold=0.7, iou_threshold=0.5, batch_size=stack, resize_image=resize_factor!=1.0)


    def get_face_from_video(filename, fileboxes, stack, length, skip, num = 0):

        # Video readers
        video_cap = cv2.VideoCapture(filename)

        if length < 1:
            
            length = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            assert length > 0

        if num > 0:

            if num > length: # Questo l'ha aggiunto
                num = length # Cozis

            step = length // num

        else:

            step = 1

        images         = list()
        all_image_inds = list()
        all_boxes      = list()
        all_scores     = list()
        all_points     = list()
        all_embeddings = list()
        for frame_count in tqdm(range(skip)):
            image = video_cap.read()[1]
        with torch.no_grad():
            for frame_count in tqdm(range(skip, length)):
                image = video_cap.read()[1]
                if (frame_count-skip)%step!=0:
                    continue
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                if len(images)==stack or frame_count==length-1:
                    # convert to torch
                    images_torch = (torch.from_numpy(np.stack(images).copy()).permute(0,3,1,2).float().to(device) - 128) / 128.0
                    images_offset = frame_count+1-len(images)
                    images = list()

                    # face detector
                    image_inds, scores, boxes, points = detect_faces(images_torch)

                    if len(boxes) > 0:
                        # extract face
                        image_inds = image_inds + images_offset

                        all_image_inds.append(image_inds.cpu())
                        all_boxes.append(boxes.cpu())
                        all_scores.append(scores.cpu())
                        all_points.append(points.cpu())

                        del image_inds
                        del boxes
                        del scores
                        del points

        video_cap.release()
        print('ok')

        image_inds = torch.cat(all_image_inds, dim=0).numpy()
        boxes      = torch.cat(all_boxes     , dim=0).numpy()
        scores     = torch.cat(all_scores    , dim=0).numpy()
        points     = torch.cat(all_points    , dim=0).numpy()

        np.savez(fileboxes, image_inds=image_inds, boxes=boxes, scores=scores, points=points)

    os.makedirs(os.path.dirname(fileboxes), exist_ok = True)
    get_face_from_video(filevideo, fileboxes, stack=stack, length=length, skip=skip, num=num)
