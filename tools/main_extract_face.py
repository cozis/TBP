import torch
import numpy as np
import pandas as pd
import os
import cv2
import time
from PIL import Image
from tqdm import tqdm
import sys
from retinaface.boxes_utils import convert_to_square_margin, roi_linear

def main_extract_face(filevideo, fileboxes, fileoutput, margin_zoom = 30, gpu = 0):

    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu"  , type=int, default=0, help="gpu id")
    parser.add_argument("-i", "--input", type=str, help="input video")
    parser.add_argument("-b", "--boxes", type=str, help="input file with box")
    parser.add_argument("-o", "--output", type=str, help="output folder")
    parser.add_argument("-m", "--margin_zoom", type=int, default=30, help="margin zoom")
    args = parser.parse_args()
    '''

    device = torch.device('cuda:%d' % int(gpu) if torch.cuda.is_available() and int(gpu) >= 0 else 'cpu')
    pathfaceout = os.path.join(fileoutput, '%s_%06d.png')

    print('Running on device: {}'.format(device))
    print('input : {}'.format(filevideo))
    print('boxes : {}'.format(fileboxes))
    print('output: {}'.format(pathfaceout))
    os.makedirs(fileoutput, exist_ok=True)

    dat = np.load(fileboxes)
    image_inds = dat['image_inds']
    boxes      = dat['boxes']
    del dat

    def extract_face(image, box, mode='edge'):
        pxb = int(box[0]); pxe = int(box[2])+1
        pyb = int(box[1]); pye = int(box[3])+1
        dx  = image.shape[1]; dy = image.shape[0]
        if len(image.shape)==3:
            face = image[max(pyb,0):min(pye,dy), max(pxb,0):min(pxe,dx), :]
            if mode is not None:
                face = np.pad(face, ((max(-pyb,0), max(pye-dy,0)), (max(-pxb,0), max(pxe-dx,0)), (0,0) ), mode=mode)
        elif len(image.shape)==2:
            face = image[max(pyb,0):min(pye,dy), max(pxb,0):min(pxe,dx)]
            if mode is not None:
                face = np.pad(face, ((max(-pyb,0), max(pye-dy,0)), (max(-pxb,0), max(pxe-dx,0)) ), mode=mode)
        else:
            assert False
        return face


    num_frames = int(np.max(image_inds)+1)

    # Video readers
    video_cap = cv2.VideoCapture(filevideo)
    for index in tqdm(range(num_frames)):
        video_cap.grab()
        keep = (image_inds==index)
        if not np.any(keep):
            continue
        boxf = boxes[keep]
        del keep
        _, image = video_cap.retrieve()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_inds1 = torch.zeros(len(boxf))
        scores = np.abs((boxf[:,2]-boxf[:,0]+1)*(boxf[:,3]-boxf[:,1]+1))

        ind = int(np.argmax(scores))
        if scores[ind]<0.25:
            print('skip', index, scores[ind].item())
            continue

        box  = boxf[ind]
        boxf = convert_to_square_margin(torch.from_numpy(boxf[ind:ind+1]), margin_zoom)[0]

        filefaceout = pathfaceout % ('face', index)
        fileboxeout = (pathfaceout % ('box', index))[:-4]+'.npy'

        face = extract_face(image, boxf, mode='edge')
        Image.fromarray(face).save(filefaceout)
        #np.save(fileboxeout, box)
