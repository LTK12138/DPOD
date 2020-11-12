import os
import re

import cv2
import numpy as np
import torch

from helper import load_obj

fx = 572.41140
px = 325.26110
fy = 573.57043
py = 242.04899
intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

list_all_images = []
for root, dirs, files in os.walk('./LineMOD_Dataset/iron/ground_truth/IDmasks/'):
    for file in files:
        if file.endswith(".png"):
            list_all_images.append(os.path.join(root, file))

# initial 6D pose prediction
regex = re.compile(r'\d+')
outliers = 0
for i in range(len(list_all_images)):
    img_adr = list_all_images[i]
    idx = regex.findall(os.path.split(img_adr)[1])[0]
    
    id_adr = './LineMOD_Dataset/iron/ground_truth/IDmasks/color' + str(idx) + '.png'
    u_adr = './LineMOD_Dataset/iron/ground_truth/Umasks/color' + str(idx) + '.png'
    v_adr = './LineMOD_Dataset/iron/ground_truth/Vmasks/color' + str(idx) + '.png'
    
    idmask_pred = cv2.imread(id_adr, cv2.IMREAD_GRAYSCALE)
    umask_pred = cv2.imread(u_adr, cv2.IMREAD_GRAYSCALE)
    vmask_pred = cv2.imread(v_adr, cv2.IMREAD_GRAYSCALE)
    # convert the masks to 240,320 shape
    temp = torch.tensor(cv2.resize(idmask_pred, (320,240), interpolation=cv2.INTER_AREA))
    upred = torch.tensor(cv2.resize(umask_pred, (320,240), interpolation=cv2.INTER_AREA))
    vpred = torch.tensor(cv2.resize(vmask_pred, (320,240), interpolation=cv2.INTER_AREA))
    coord_2d = (temp == 11).nonzero(as_tuple=True)

    adr = "./LineMOD_Dataset/iron/predicted_pose/info_" + str(idx) + ".txt"

    coord_2d = torch.cat((coord_2d[0].view(
        coord_2d[0].shape[0], 1), coord_2d[1].view(coord_2d[1].shape[0], 1)), 1)
    uvalues = upred[coord_2d[:, 0], coord_2d[:, 1]]
    vvalues = vpred[coord_2d[:, 0], coord_2d[:, 1]]

    dct_keys = torch.cat((uvalues.view(-1, 1), vvalues.view(-1, 1)), 1)
    dct_keys = tuple(dct_keys.numpy())
    dct = load_obj("./LineMOD_Dataset/iron/UV-XYZ_mapping")
    mapping_2d = []
    mapping_3d = []
    for count, (u, v) in enumerate(dct_keys):
        if (u, v) in dct:
            mapping_2d.append(np.array(coord_2d[count]))
            mapping_3d.append(dct[(u, v)])
    # Get the 6D pose from rotation and translation matrices
    # PnP needs atleast 6 unique 2D-3D correspondences to run
    if len(mapping_2d) >= 4 or len(mapping_3d) >= 4:
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d, dtype=np.float32),
                                                        np.array(mapping_2d, dtype=np.float32), intrinsic_matrix, distCoeffs=None,
                                                        iterationsCount=150, reprojectionError=1.0, flags=cv2.SOLVEPNP_P3P)
        rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
        rot[np.isnan(rot)] = 1
        tvecs[np.isnan(tvecs)] = 1
        tvecs = np.where(-100 < tvecs, tvecs, np.array([-100.]))
        tvecs = np.where(tvecs < 100, tvecs, np.array([100.]))
        rot_tra = np.append(rot, tvecs, axis=1)
        # save the predicted pose
        np.savetxt(adr, rot_tra)
    else:  # save a pose full of zeros
        outliers += 1
        rot_tra = np.ones((3, 4))
        rot_tra[:, 3] = 0
        np.savetxt(adr, rot_tra)
print("Number of instances where PnP couldn't be used: ", outliers)
