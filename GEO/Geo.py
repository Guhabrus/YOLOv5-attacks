import numpy as np
from math import atan, cos, sin, tan
from cameraLocations import cameras

viewAngleHorizontal = 150
viewAngleVertical = 180
heightAboveGround = 40 * 0.0001 # метра

imageWidth = 1920
imageHeight = 1080

def geoToList(latlon):
    return np.array((latlon['lat'], latlon['lng']))

def listToGeo(latlon):
    return {'lat': latlon[0], 'lng': latlon[1] }


def getGeoCoordinates(A, B, C, D, X, Y):
    A, B, C, D = list(map(geoToList, [A, B, C, D]))
    vBC = (C - B) / imageHeight
    vAD = (D - A) / imageHeight
    latlonPixel1 = vBC * (imageHeight - Y) + B
    latlonPixel2 = vAD * (imageHeight - Y) + A
    vM = (latlonPixel2 - latlonPixel1) / imageWidth
    M = vM * X + latlonPixel1
    # print(f"M {M}")
    return listToGeo(M)



def computeImagePoint(alpha, beta, x, y, h, O):
    O_x, O_y = O
    len_M = h * tan(alpha - viewAngleVertical * ((imageHeight - y) / imageHeight - 0.5))
    M_x = O_x - cos(beta - viewAngleHorizontal * ((imageWidth - x) / imageWidth - 0.5)) * len_M
    M_y = O_y - sin(beta - viewAngleHorizontal * ((imageWidth - x) / imageWidth - 0.5)) * len_M
    return np.array([M_x, M_y])


def getTrapeziumPoints(cameraNumber):
    O = geoToList(cameras[cameraNumber]['coordinates'])
    
    angle = cameras[cameraNumber]['angle'][0]
    alpha = angle['alpha']
    beta  = angle['beta']
    
    A = computeImagePoint(alpha, beta, 0, imageHeight, heightAboveGround, O)
    B = computeImagePoint(alpha, beta, imageWidth, imageHeight, heightAboveGround, O)
    C = computeImagePoint(alpha, beta, imageWidth, 0, heightAboveGround, O)
    D = computeImagePoint(alpha, beta, 0, 0, heightAboveGround, O)
    print(f"A - {A}")
    return [listToGeo(A), listToGeo(B), listToGeo(C), listToGeo(D)]


def get_coords(X, Y):
    Kam = getTrapeziumPoints(1)
    # centre = cameras[1]['coordinates']
    res = getGeoCoordinates(Kam[0],Kam[1],Kam[2],Kam[3], X,Y )
    # print(f'ass - {res}')
    return res

# coordinates - координаты камеры
#center - координты точки для проверки
# lat - широта
# lng долгота



# get_coords()