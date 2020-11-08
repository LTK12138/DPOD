import math
import os
import random
from vtk_renderer import VTKRenderer

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Rx(theta):
  return np.matrix([[1, 0, 0],
                    [0, math.cos(theta), -math.sin(theta)],
                    [0, math.sin(theta), math.cos(theta)]])

def Ry(theta):
  return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                    [0, 1, 0],
                    [-math.sin(theta), 0, math.cos(theta)]])

def Rz(theta):
  return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0],
                    [0, 0, 1]])

def get_rot(z, x):
    return Rx(x) * Rz(z)

def fill_holes(idmask, umask, vmask):
    """
    Helper function to fill the holes in id , u and vmasks
        Args:
                idmask (np.array): id mask whose holes you want to fill
        umask (np.array): u mask whose holes you want to fill
        vmask (np.array): v mask whose holes you want to fill
        Returns:
                filled_id_mask (np array): id mask with holes filled
        filled_u_mask (np array): u mask with holes filled
        filled_id_mask (np array): v mask with holes filled
    """
    idmask = np.array(idmask, dtype='float32')
    umask = np.array(umask, dtype='float32')
    vmask = np.array(vmask, dtype='float32')
    thr, im_th = cv2.threshold(idmask, 0, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    res = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
    im_th = cv2.bitwise_not(im_th)
    des = cv2.bitwise_not(res)
    mask = np.array(des-im_th, dtype='uint8')
    filled_id_mask = cv2.inpaint(idmask, mask, 5, cv2.INPAINT_TELEA)
    filled_u_mask = cv2.inpaint(umask, mask, 5, cv2.INPAINT_TELEA)
    filled_v_mask = cv2.inpaint(vmask, mask, 5, cv2.INPAINT_TELEA)

    return filled_id_mask, filled_u_mask, filled_v_mask

def generate_sample(root_dir, background_dir, intrinsic_matrix, classes, nSampleAng=5, nSampleD=50, nSampleN=20):
    for label in classes.keys():
        base_path = root_dir + label + "/gen/"
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        renderer = VTKRenderer(640, 480, intrinsic_matrix, root_dir + label + "/mesh.ply", ".\\val2017\\a.jpg")

        # Read point Point Cloud Data
        ptcld_file = root_dir + label + "/object.xyz"
        pt_cld_data = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
        ones = np.ones((pt_cld_data.shape[0], 1))
        homogenous_coordinate = np.append(pt_cld_data[:, :3], ones, axis=1)

        # for d in range(70, 120, nSampleD):
        for zAngle in range(360//nSampleAng):
            for xAngle in range(180//nSampleAng+1):
                rot_matrix = get_rot(zAngle*math.radians(nSampleAng), xAngle*math.radians(nSampleAng)-math.pi/2)
                trans_matrix = np.array([random.random()*70-35,random.random()*50-25,100]).reshape((3,1))
                rigid_transformation = np.append(rot_matrix, trans_matrix, axis=1)

                ID_mask = np.zeros((480, 640))
                U_mask = np.zeros((480, 640))
                V_mask = np.zeros((480, 640))

                # Perspective Projection to obtain 2D coordinates for masks
                homogenous_2D = intrinsic_matrix @ (
                    rigid_transformation @ homogenous_coordinate.T)
                coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
                coord_2D = ((np.floor(coord_2D)).T).astype(int)
                x_2d = np.clip(coord_2D[:, 0], 0, 639)
                y_2d = np.clip(coord_2D[:, 1], 0, 479)
                ID_mask[y_2d, x_2d] = classes[label]

                # Generate Ground Truth UV Maps
                centre = np.mean(pt_cld_data, axis=0)
                length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                                    pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
                unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                                1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
                U = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
                V = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
                U_mask[y_2d, x_2d] = U.reshape(U_mask[y_2d, x_2d].shape)
                V_mask[y_2d, x_2d] = V.reshape(V_mask[y_2d, x_2d].shape)

                fileName = str(zAngle) + "_" + str(xAngle) + ".png"

                # Saving ID, U and V masks after using the fill holes function
                ID_mask, U_mask, V_mask = fill_holes(ID_mask, U_mask, V_mask)
                cv2.imwrite(base_path + "id" + fileName, ID_mask)
                cv2.imwrite(base_path + "u" + fileName, U_mask * 255)
                cv2.imwrite(base_path + "v" + fileName, V_mask * 255)

                # RGB
                background_img_adr = background_dir + random.choice(os.listdir(background_dir))
                rgb = cv2.imread(background_img_adr)
                rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_AREA)

                rgb = renderer.getImage(rigid_transformation)
                # cv2.imshow("gen", rgb)
                # cv2.waitKey(0)
                cv2.imwrite(base_path + "rgb" + fileName, rgb)
