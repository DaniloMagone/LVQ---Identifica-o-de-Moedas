import csv
import cv2 as cv
import numpy as np
import glob
import os

def get_features(filename):
    cvimg = cv.imread(filename)
    ycrcb = cv.cvtColor(cvimg, cv.COLOR_BGR2YCrCb)
    radius = cvimg.shape[0] >> 1 # Característica 1: raio
    center = np.array([radius, radius])
    # Iterando cores
    innerCircleMaxRad = int(0.5 * radius)
    ringRadiuses = (int(0.75 * radius), int(0.9 * radius))
    fullCircleColors = []
    innerCircleColors = []
    ringColors = []
    rows, columns, _ = ycrcb.shape
    for i in range(rows):
        for j in range(columns):
            dist = np.linalg.norm(np.array([i, j]) - center)
            if dist > radius:
                continue
            colors = ycrcb[i, j, 1:]
            fullCircleColors.append(colors)
            if dist <= innerCircleMaxRad:
                innerCircleColors.append(colors)
            if ringRadiuses[0] <= dist <= ringRadiuses[1]:
                ringColors.append(colors)
    # Cor média e diferença de cor média entre coroa e círculo interno
    fullCircleAvg = np.mean(np.asarray(fullCircleColors), axis=0)
    innerCircleAvg = np.mean(np.asarray(innerCircleColors), axis=0)
    ringAvg = np.mean(np.asarray(ringColors), axis=0)
    return [
        fullCircleAvg[0], fullCircleAvg[1],
        abs(innerCircleAvg[0] - ringAvg[0]),
        abs(innerCircleAvg[1] - ringAvg[1]),
        radius
    ]
    


def write_dataset(filename, images_dir):
    with open(filename, 'w', newline='') as fd:
        writer = csv.writer(fd)
        for filename in sorted(glob.glob(f'{images_dir}/*.bmp')):
            name = os.path.splitext(filename)[0]
            cluster = name.split('_')[1]
            features = get_features(filename)
            featuresFmt = ', '.join([f'{v:.2f}' for v in features])
            print(f'{filename} : {featuresFmt}')
            writer.writerow(get_features(filename) + [cluster])


if __name__ == '__main__':
    write_dataset('init.csv', 'init')
    write_dataset('training.csv', 'train')



