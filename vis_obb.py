
import re
import cv2
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from helper import save_obj, load_obj, create_bounding_box
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if not os.path.exists("./LineMOD_Dataset/iron/obb/"):
    os.mkdir("./LineMOD_Dataset/iron/obb/")

# Intrinsic Parameters of the Camera
fx = 572.41140
px = 325.26110
fy = 573.57043
py = 242.04899
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

# Read Point Point Cloud Data
ptcld_file = "./LineMOD_Dataset/iron/object.xyz"
pt_cld_data = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
ones = np.ones((pt_cld_data.shape[0], 1))
homogenous_coordinate = np.append(pt_cld_data[:, :3], ones, axis=1)

regex = re.compile(r'\d+')
datpath = "./LineMOD_Dataset/iron/predicted_pose/"
for dat_adr in [f for f in os.listdir(datpath) if os.path.isfile(os.path.join(datpath, f))]:
    pred_pose = np.loadtxt(datpath + dat_adr)
    idx = regex.findall(dat_adr)[0]
    image = cv2.imread("./LineMOD_Dataset/iron/changed_background/color" + idx + ".png")
    image = create_bounding_box(image, pred_pose, pt_cld_data, intrinsic_matrix)

    if image is not None:
        cv2.imwrite("./LineMOD_Dataset/iron/obb/" + idx + ".png", image)