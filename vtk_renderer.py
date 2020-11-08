# https://gist.github.com/decrispell/fc4b69f6bedf07a3425b
# https://gist.github.com/benoitrosa/ffdb96eae376503dba5ee56f28fa0943
import cv2
import numpy as np
import vtk
from vtkmodules.util import numpy_support

class VTKRenderer():
    def __init__(self, w, h, K, fileName, bgFileName):
        self.w = w
        self.h = h

        self.init_vtk(fileName, bgFileName)
        self.init_camera(K)
        self.create_lights()

    def getImage(self, pose):
        pose = np.append(pose, [[0, 0, 0, 1]], axis=0)
        poseT = vtk.vtkTransform()
        poseT.SetMatrix(np.array(pose.flatten()).squeeze())
        self.actor.SetUserTransform(poseT)

        # Render the scene into a numpy array for openCV processing
        self.renWin.Render()
        winToIm = vtk.vtkWindowToImageFilter()
        winToIm.SetInput(self.renWin)
        winToIm.SetInputBufferTypeToRGBA()
        winToIm.ReadFrontBufferOff()
        winToIm.Update()

        image = winToIm.GetOutput()
        w, h, _ = image.GetDimensions()
        dat = image.GetPointData().GetScalars()
        c = dat.GetNumberOfComponents()
        arr = cv2.flip(numpy_support.vtk_to_numpy(dat).reshape(h, w, c), 0)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    def init_vtk(self, fileName, bgFileName):
        self.reader = vtk.vtkPLYReader()
        self.reader.SetFileName(fileName)
        self.reader.Update()
        self.readerScaleTransform = vtk.vtkTransform()
        self.readerScaleTransform.Scale(0.1, 0.1, 0.1)
        self.readerScaleTransformFilter = vtk.vtkTransformPolyDataFilter()
        self.readerScaleTransformFilter.SetInputConnection(
            self.reader.GetOutputPort())
        self.readerScaleTransformFilter.SetTransform(self.readerScaleTransform)
        self.readerScaleTransformFilter.Update()
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(
            self.readerScaleTransformFilter.GetOutputPort())
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)

        self.bgreader = vtk.vtkJPEGReader()
        self.bgreader.SetFileName(bgFileName)
        self.bgreader.Update()
        self.bgactor = vtk.vtkImageActor()
        imageData = self.bgreader.GetOutput()
        self.bgactor.SetInputData(imageData)

        self.bgrenderer = vtk.vtkRenderer()
        self.bgrenderer.AddActor(self.bgactor)
        self.bgrenderer.ResetCamera()
        bgcamera = self.bgrenderer.GetActiveCamera()
        bgcamera.ParallelProjectionOn()

        origin = imageData.GetOrigin()
        spacing = imageData.GetSpacing()
        extent = imageData.GetExtent()
        xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
        yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
        yd = (extent[3] - extent[2] + 1)*spacing[1]
        d = bgcamera.GetDistance()
        bgcamera.SetParallelScale(0.5 * yd)
        bgcamera.SetFocalPoint(xc, yc, 0.0)
        bgcamera.SetPosition(xc, yc, d)

        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(0, 1, 0)
        self.renderer.ResetCamera()

        self.renWin = vtk.vtkRenderWindow()
        self.renWin.SetSize(self.w, self.h)
        self.renWin.SetOffScreenRendering(1)
        self.renWin.SetNumberOfLayers(2)

        self.bgrenderer.SetLayer(0)
        self.bgrenderer.InteractiveOff()
        self.renderer.SetLayer(1)
        self.renWin.AddRenderer(self.bgrenderer)
        self.renWin.AddRenderer(self.renderer)

    def init_camera(self, K):
        self.K = K  # Camera matrix

        self.f = np.array([K[0, 0], K[1, 1]])  # focal lengths
        self.c = K[:2, 2]  # principal point

        # Set basic camera parameters in VTK
        cam = self.renderer.GetActiveCamera()
        near = 0.1
        far = 1000.0
        cam.SetClippingRange(near, far)

        # Position is at origin, looking in z direction with y down
        cam.SetPosition(0, 0, 0)
        cam.SetFocalPoint(0, 0, 1)
        cam.SetViewUp(0, -1, 0)

        # Set window center for offset principal point
        wcx = -2.0*(self.c[0] - self.w / 2.0) / self.w
        wcy = 2.0*(self.c[1] - self.h / 2.0) / self.h
        cam.SetWindowCenter(wcx, wcy)

        # Set vertical view angle as a indirect way of setting the y focal distance
        angle = 180 / np.pi * 2.0 * np.arctan2(self.h / 2.0, self.f[1])
        cam.SetViewAngle(angle)

        # Set the image aspect ratio as an indirect way of setting the x focal distance
        m = np.eye(4)
        aspect = self.f[1] / self.f[0]
        m[0, 0] = 1.0 / aspect
        t = vtk.vtkTransform()
        t.SetMatrix(m.flatten())
        cam.SetUserTransform(t)

    def create_lights(self):
        thetas = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

        self.lights = []
        self.lightPos = []

        for i in range(3):
            xp = np.cos(thetas[i])
            yp = np.sin(thetas[i])
            zp = 0

            self.lightPos.append(np.array([xp, yp, zp]))

            self.lights.append(vtk.vtkLight())
            self.lights[i].SetPosition(xp, yp, zp)
            self.lights[i].SetFocalPoint(0, 0, 0)
            self.renderer.AddLight(self.lights[i])

            
