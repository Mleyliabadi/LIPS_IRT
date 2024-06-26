#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gmsh
import numpy as np

def GenerateWheelMeshFileStandardVersion(outputFile, wheelDimensions, meshSize):
    """PYO code using gmsh to generate a mesh"""
    gmsh.initialize()
    model_name=outputFile
    gmsh.model.add(model_name)
    cm = 1.
    center_x = 0 * cm
    center_y = 15 * cm
    rayon_1, rayon_2 = wheelDimensions
    Lc1 = meshSize
    Lc2 = 0.01
    # We start by defining some points and some lines. To make the code shorter we
    # can redefine a namespace:
    factory = gmsh.model.occ
    factory.addPoint(center_x, center_y, 0, Lc1, 1)
    # Circles
    factory.addCircle(center_x, center_y, 0, rayon_1, 10)  # instead of addCircleArc(2, 1, 2, 10)
    factory.addCircle(center_x, center_y, 0, rayon_2, 11)
    factory.addCurveLoop([10], 20)
    factory.addCurveLoop([11], 21)
    # Add surfaces
    factory.addPlaneSurface([21, 20], 30)
    factory.synchronize()
    gmsh.model.addPhysicalGroup(2, [30], 100)
    # Assign a mesh size to all the points:
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), Lc1)
    # gmsh.model.mesh.setSize(gmsh.model.getBoundary([31], False, False, True), Lc2)
    gmsh.model.mesh.setAlgorithm(2, 30, 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.msh" % model_name)
    gmsh.finalize()

class Standard3DWheelGenerator:
    def __init__(self,wheel_Dimensions,mesh_size):
        self.name = "Standard 3D Wheel"
        self.wheel_Dimensions=wheel_Dimensions
        self.center=(0,wheel_Dimensions[1])
        self.mesh_size=mesh_size
        self.tagMap={"Interior":2,"Exterior":3}

    def GenerateMesh(self,outputFile):
        wheelGeometry=self._CreateWheelGeometry()
        self._TagRelevantSurfaces(wheelGeometry)
        self._Generate(outputFile)
    
    def _CreateWheelGeometry(self):
        gmsh.initialize()
        gmsh.model.add(self.name)
        rayon_1, rayon_2, thickness = self.wheel_Dimensions
        center_x, center_y = self.center
        factory = gmsh.model.occ
        factory.addPoint(center_x, center_y, 0, self.mesh_size, 1)

        factory.addCircle(center_x, center_y, 0, rayon_1, 10)
        factory.addCircle(center_x, center_y, 0, rayon_2, 11)
        factory.addCurveLoop([10], 20)
        factory.addCurveLoop([11], 21)

        factory.addPlaneSurface([21, 20], 30)
        extrudedGeo=factory.extrude([[2,30]], 0, 0, thickness)
        factory.synchronize()
        return extrudedGeo

    def _TagRelevantSurfaces(self,wheelGeometry):
        volume_entities = []
        surface_entities = []
        for entity in wheelGeometry:
            if entity[0] == 3:
                volume_entities.append(entity[1])
            elif entity[0] ==2:
                surface_entities.append(entity[1])
        gmsh.model.addPhysicalGroup(3, volume_entities, tag=1)
        gmsh.model.setPhysicalName(3, 1, "MeshVolume")
        rayon_1, rayon_2, thickness = self.wheel_Dimensions

        for surfaceEntity in surface_entities:
            entity=(2,surfaceEntity)
            boundingBoxEntity=gmsh.model.getBoundingBox(*entity)
            boxThickness=np.abs(boundingBoxEntity[5]-boundingBoxEntity[2])
            boxLenght=np.abs(boundingBoxEntity[3]-boundingBoxEntity[0])
            if np.abs(boxThickness-thickness)<1e-3:
                if np.abs(boxLenght-2*rayon_1)<1e-3:
                    surfaceTagName="Interior"
                elif np.abs(boxLenght-2*rayon_2)<1e-3:
                    surfaceTagName="Exterior"
                else:
                    continue
                surfaceTagValue=self.tagMap[surfaceTagName]
                gmsh.model.addPhysicalGroup(2, [surfaceEntity], tag=surfaceTagValue)
                gmsh.model.setPhysicalName(2, surfaceTagValue, surfaceTagName)

    def _Generate(self,outputFile:str):
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.mesh_size)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2)
        gmsh.write("%s.msh" % outputFile)
        gmsh.finalize()

    def SaveMeshConfig(self,fileToSave:str):
        meshData={"wheel_Dimensions":self.wheel_Dimensions,
                  "mesh_size":self.mesh_size}
        np.save(fileToSave, meshData)

class DentedWheelGenerator():
    def __init__(self,wheel_Dimensions,teeth_Size,tread_Angle_deg,mesh_size):
        self.name = "Dented 2D Wheel"
        self.wheel_Dimensions=wheel_Dimensions
        self.center=(0,wheel_Dimensions[-1])
        self.teeth_Size=teeth_Size
        self.tread_Angle_deg=tread_Angle_deg
        self.mesh_size=mesh_size

    def GenerateMesh(self,outputFile):
        radius_min,radius_inter,radius_max=self.wheel_Dimensions
        self.GenerateInternalWheel(outputFile="wheel_internal",partialWheelDimensions=(radius_min,radius_inter))
        self.GenerateExternalDentedWheel(outputFile="wheel_external",partialWheelDimensions=(radius_inter,radius_max))
        MergeMeshes(outputFile, ["wheel_external.msh", "wheel_internal.msh"])

    def GenerateInternalWheel(self,outputFile,partialWheelDimensions):
        gmsh.initialize()
        gmsh.model.add(outputFile)

        nb_motif = self.ComputeNbMotif()
        tread_Angle_rad = self.ConvertTreadAngleToRadiant()

        factory=self.CreateSingleMotif(partialWheelDimensions=partialWheelDimensions)

        out_list = []
        out = [(2, 1)]
        out_list.append(1)
        for _ in range(nb_motif):
            out = gmsh.model.occ.copy([(2, out[0][1])])
            center_x ,center_y = self.center
            gmsh.model.occ.rotate(out, center_x, center_y, 0, 0, 0, 1, 2 * tread_Angle_rad)
            out_list.append(out[0][1])

        factory.synchronize()

        # Assign a mesh size to all the points:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.mesh_size)
        gmsh.model.geo.remove([(1, 1)])
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate(2)
        gmsh.write("%s.msh" % outputFile)
        gmsh.finalize()


    def GenerateExternalDentedWheel(self,outputFile,partialWheelDimensions):
        gmsh.initialize()
        gmsh.model.add(outputFile)

        nb_motif=self.ComputeNbMotif()
        tread_Angle_rad = self.ConvertTreadAngleToRadiant()

        factory=self.CreateSingleMotif(partialWheelDimensions=partialWheelDimensions)

        out_list = []
        teeth_dx,teeth_dy=self.teeth_Size
        factory.addRectangle(-teeth_dx/2, -teeth_dy, 0, teeth_dx, 2*teeth_dy, 0)
        gmsh.model.occ.cut([(2, 1)], [(2, 0)], 3)

        out2 = [(2, 3)]
        out_list.append(3)
        for _ in range(nb_motif):
            out2 = gmsh.model.occ.copy([(2, out2[0][1])])
            center_x,center_y =self.center
            gmsh.model.occ.rotate(out2, center_x, center_y, 0, 0, 0, 1, 2 * tread_Angle_rad)
            out_list.append(out2[0][1])

        factory.synchronize()

        # Assign a mesh size to all the points:
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), self.mesh_size)
        gmsh.model.geo.remove([(1, 1)])
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate(2)
        gmsh.write("%s.msh" % outputFile)

        gmsh.finalize()

    def ComputeNbMotif(self):
        nb_motif = (360-2*self.tread_Angle_deg)/(2*self.tread_Angle_deg)
        if nb_motif.is_integer():
            nb_motif=int(nb_motif)
        else:
            raise Exception("Not an integer number of motif")
        return nb_motif 

    def ConvertTreadAngleToRadiant(self):
        return self.tread_Angle_deg * np.pi / 180

    def CreateSingleMotif(self,partialWheelDimensions):
        factory = gmsh.model.occ
        center_x, center_y=self.center
        mesh_size=self.mesh_size
        factory.addPoint(center_x, center_y, 0, mesh_size, 1)
        radius1, radius2 = partialWheelDimensions
        tread_Angle_rad = self.tread_Angle_deg * np.pi / 180 
        factory.addPoint(-np.sin(tread_Angle_rad) * radius2 + center_x,
                        -np.cos(tread_Angle_rad) * radius2 + center_y, 0, mesh_size, 2)
        factory.addPoint(np.sin(tread_Angle_rad) * radius2 + center_x,
                        -np.cos(tread_Angle_rad) * radius2 + center_y, 0, mesh_size, 3)

        factory.addPoint(-np.sin(tread_Angle_rad) * radius1 + center_x,
                        -np.cos(tread_Angle_rad) * radius1 + center_y, 0, mesh_size, 4)
        factory.addPoint(np.sin(tread_Angle_rad) * radius1 + center_x,
                        -np.cos(tread_Angle_rad) * radius1 + center_y, 0, mesh_size, 5)

        factory.addCircleArc(2, 1, 3, 1)
        factory.addCircleArc(4, 1, 5, 2)

        factory.addLine(4, 2, 3)
        factory.addLine(3, 5, 4)

        factory.addCurveLoop([2, 3, 1, 4], 1)
        factory.addPlaneSurface([1], 1)
        return factory

    def SaveMeshConfig(self,fileToSave:str):
        meshData={"wheel_Dimensions":self.wheel_Dimensions,
                  "teeth_Size":self.teeth_Size,
                  "tread_Angle_deg":self.tread_Angle_deg,
                  "mesh_size":self.mesh_size}
        np.save(fileToSave, meshData)

def LoadMeshConfig(fileToLoad:str):
    meshData = np.load(fileToLoad,allow_pickle='TRUE').item()
    return meshData


def GenerateWheelMeshFileTreadVersion(outputFile, wheelDimensions, meshSize):
    """05/07/2021 new code to generate a simple wheel in a sectorial manner"""
    gmsh.initialize()
    gmsh.model.add(outputFile)

    mesh_size = meshSize  # meshSize # 0.7
    tread_angle = 20 * np.pi / 180

    cm = 1.
    center_x = 0 * cm
    center_y = 15 * cm
    # center_y = 15 * cm
    radius1, radius2 = wheelDimensions
    layers_number = 8

    factory = gmsh.model.occ
    factory.addPoint(center_x, center_y, 0, mesh_size, 1)
    factory.addPoint(-np.sin(tread_angle) * radius2 + center_x,
                     -np.cos(tread_angle) * radius2 + center_y, 0, mesh_size, 2)
    factory.addPoint(np.sin(tread_angle) * radius2 + center_x,
                     -np.cos(tread_angle) * radius2 + center_y, 0, mesh_size, 3)

    factory.addPoint(-np.sin(tread_angle) * radius1 + center_x,
                     -np.cos(tread_angle) * radius1 + center_y, 0, mesh_size, 4)
    factory.addPoint(np.sin(tread_angle) * radius1 + center_x,
                     -np.cos(tread_angle) * radius1 + center_y, 0, mesh_size, 5)

    factory.addCircleArc(2, 1, 3, 1)
    factory.addCircleArc(4, 1, 5, 2)

    factory.addLine(4, 2, 3)
    factory.addLine(3, 5, 4)

    factory.addCurveLoop([2, 3, 1, 4], 1)
    factory.addPlaneSurface([1], 1)

    out_list = []
    out = [(2, 1)]
    out_list.append(1)
    for t in range(0, layers_number):
        out = gmsh.model.occ.copy([(2, out[0][1])])
        gmsh.model.occ.rotate(out, center_x, center_y, 0, 0, 0, 1, 2 * tread_angle)
        out_list.append(out[0][1])

    factory.synchronize()

    # Assign a mesh size to all the points:
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.geo.remove([(1, 1)])
    #
    # # gmsh.model.mesh.setSize(gmsh.model.getBoundary([31], False, False, True), Lc2)
    # # gmsh.model.mesh.setAlgorithm(2, 30, 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.msh" % outputFile)

    # Launch the GUI to see the results:
    # import sys
    # if '-nopopup' not in sys.argv:
    #     gmsh.fltk.run()

    gmsh.finalize()



def GenerateWheelMeshFileAsymVersion(outputFile, wheelDimensions, meshSize, refMesh):
    gmsh.initialize()
    gmsh.model.add(outputFile)

    mesh_size = 2.#meshSize # 0.7
    tread_angle = 20 * np.pi / 180

    cm = 1.
    center_x = 0 * cm
    center_y = 50 * cm
    radius1, radius2 = wheelDimensions
    layers_number = 8

    factory = gmsh.model.occ
    factory.addPoint(center_x, center_y, 0, mesh_size, 1)
    factory.addPoint(-np.sin(tread_angle) * radius2 + center_x,
                     -np.cos(tread_angle) * radius2 + center_y, 0, mesh_size, 2)
    factory.addPoint(np.sin(tread_angle) * radius2 + center_x,
                     -np.cos(tread_angle) * radius2 + center_y, 0, mesh_size, 3)

    factory.addPoint(-np.sin(tread_angle) * radius1 + center_x,
                     -np.cos(tread_angle) * radius1 + center_y, 0, mesh_size, 4)
    factory.addPoint(np.sin(tread_angle) * radius1 + center_x,
                     -np.cos(tread_angle) * radius1 + center_y, 0, mesh_size, 5)

    factory.addCircleArc(2, 1, 3, 1)
    factory.addCircleArc(4, 1, 5, 2)

    factory.addLine(4, 2, 3)
    factory.addLine(3, 5, 4)

    factory.addCurveLoop([2, 3, 1, 4], 1)
    factory.addPlaneSurface([1], 1)

    out_list = []
    if refMesh is None:
        factory.addRectangle(-2, -10, 0, 4, 20, 0)
        gmsh.model.occ.cut([(2, 1)], [(2, 0)], 3)


        out2 = [(2, 3)]
        out_list.append(3)
        for t in range(0, layers_number):
            out2 = gmsh.model.occ.copy([(2, out2[0][1])])
            gmsh.model.occ.rotate(out2, center_x, center_y, 0, 0, 0, 1, 2 * tread_angle)
            out_list.append(out2[0][1])

    else:
        out = [(2, 1)]
        out_list.append(1)
        for t in range(0, layers_number):
            out = gmsh.model.occ.copy([(2, out[0][1])])
            gmsh.model.occ.rotate(out, center_x, center_y, 0, 0, 0, 1, 2 * tread_angle)
            out_list.append(out[0][1])

    factory.synchronize()

    # Assign a mesh size to all the points:
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.geo.remove([(1, 1)])
        #
        # # gmsh.model.mesh.setSize(gmsh.model.getBoundary([31], False, False, True), Lc2)
        # # gmsh.model.mesh.setAlgorithm(2, 30, 2)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.msh" % outputFile)

    gmsh.finalize()

    return 0


def MergeMeshes(outputFile, meshFiles):
    gmsh.initialize()
    model_name = outputFile
    gmsh.model.add(model_name)
    for mesh in meshFiles:
        gmsh.merge(mesh)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.msh" % outputFile)

    gmsh.finalize()


def GenerateCoincidentHFLFMeshes(romMeshFile, refMeshFile, interRadius, wheelDim, meshSize, version="Standard"):
    wheel_radius = interRadius
    if version=="Asym":
        GenerateWheelMeshFileAsymVersion(outputFile=romMeshFile, wheelDimensions=(wheel_radius, wheelDim[1]), meshSize=meshSize, refMesh=None)
        GenerateWheelMeshFileAsymVersion(outputFile="wheel_inter", wheelDimensions=(wheelDim[0], wheel_radius), meshSize=meshSize, refMesh=1)
    elif version=="Tread":
        GenerateWheelMeshFileTreadVersion(outputFile=romMeshFile, wheelDimensions=(wheel_radius, wheelDim[1]), meshSize=meshSize)
        GenerateWheelMeshFileTreadVersion(outputFile="wheel_inter", wheelDimensions=(wheelDim[0], wheel_radius), meshSize=meshSize)
    elif version=="Standard":
        GenerateWheelMeshFileStandardVersion(outputFile=romMeshFile, wheelDimensions=(wheel_radius, wheelDim[1]), meshSize=meshSize)
        GenerateWheelMeshFileStandardVersion(outputFile="wheel_inter", wheelDimensions=(wheelDim[0], wheel_radius), meshSize=meshSize)

    MergeMeshes(refMeshFile, [romMeshFile+".msh", "wheel_inter.msh"])

def GenerateUnitSquareMesh(outputFile, meshSize):
    gmsh.initialize()
    factory = gmsh.model.occ
    factory.addRectangle(0, 0, 0, 1, 1, 1)
    factory.synchronize()
    gmsh.model.mesh.setTransfiniteSurface(1)
    gmsh.option.setNumber("Mesh.MeshSizeMax", meshSize)
    gmsh.model.mesh.generate()
    gmsh.write("%s.msh" % outputFile)

    gmsh.finalize()


def CheckIntegrity_DentedWheel():    
    meshConfig={"wheel_Dimensions":(30.,36.,40.),
                "teeth_Size":(10/3.0,10/6.0),
                "tread_Angle_deg":5.0,
                "mesh_size":1.0}
    myDentedWheel=DentedWheelGenerator(**meshConfig)
    myDentedWheel.GenerateMesh(outputFile="DentedWheelRevised")

    fileToSaveConfig="MyMeshConfig.npy"
    myDentedWheel.SaveMeshConfig(fileToSaveConfig)
    myMeshConfig=LoadMeshConfig(fileToSaveConfig)
    assert(myMeshConfig==meshConfig)
    return "ok"

def CheckIntegrity_3DStandardWheel():
    mesh3DConfig={"wheel_Dimensions":(8.,15.,4),
                "mesh_size":0.75}
    my3DWheel=Standard3DWheelGenerator(**mesh3DConfig)
    my3DWheel.GenerateMesh(outputFile="my3DWheel")

    fileToSaveConfig="My3DMeshConfig.npy"
    my3DWheel.SaveMeshConfig(fileToSaveConfig)
    myMeshConfig=LoadMeshConfig(fileToSaveConfig)
    assert(myMeshConfig==mesh3DConfig)
    return "ok"

def CheckIntegrity_CoincidentWheels():
    wheelDimensions=(8.,15.)
    GenerateCoincidentHFLFMeshes(romMeshFile="wheel_romStandard",refMeshFile="wheel_refStandard",interRadius=11.5,wheelDim=wheelDimensions,meshSize=1.0,version="Standard")
    GenerateCoincidentHFLFMeshes(romMeshFile="wheel_romTread",refMeshFile="wheel_refTread",interRadius=11.5,wheelDim=wheelDimensions,meshSize=1.0,version="Tread")
    wheelDimensions=(20.,50.)
    GenerateCoincidentHFLFMeshes(romMeshFile="wheel_romAsym",refMeshFile="wheel_refAsym",interRadius=39.0,wheelDim=wheelDimensions,meshSize=1.0,version="Asym")
    GenerateUnitSquareMesh(outputFile="UnitSquare",meshSize=1)
    return "ok"

def CheckIntegrity():

    totest = [
        CheckIntegrity_CoincidentWheels,
        CheckIntegrity_DentedWheel,
        CheckIntegrity_3DStandardWheel
              ]

    for test in totest:
        res =  test()
        if  res.lower() != "ok" :
            return res

    return "OK"

if __name__ =="__main__":
    CheckIntegrity()