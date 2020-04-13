import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from imutils import contours
import shutil
from time import time

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import argparse
from pytorchModel import Net

from PIL import Image
import pytesseract

import codecs
import json

from warnings import filterwarnings
filterwarnings('ignore')


CAT_MAPPING = {0: 'both', 1: 'double_text', 2: 'down', 3: 'inverse_arrow', 4: 'other', 5: 'right', 6: 'single_text'}

def calcCoordinates(line):
    rho, theta = line.reshape(-1)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 100000 * (-b))
    y1 = int(y0 + 100000 * (a))
    x2 = int(x0 - 100000 * (-b))
    y2 = int(y0 - 100000 * (a))

    return x1, y1, x2, y2


def calcTangent(x1, y1, x2, y2):
    if (y2 - y1) != 0:
        return abs((x2 - x1) / (y2 - y1))
    else:
        return 1000


def getLines(image, filter=True):
    '''
    The method finds each sudoku cell with Lines
    '''

    # apply some preprocessing before applying Hough transform
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 90, 150, apertureSize=3)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=1)

    # find lines on the preprocessed image u|sing Hough Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 400)

    if not lines.any():
        print('No lines were found')
        exit()

    # calculate how many horizontal lines were found
    tot = 0
    for line in lines:
        x1, y1, x2, y2 = calcCoordinates(line)

        tan = calcTangent(x1, y1, x2, y2)
        if tan > 1000:
            tot += 1

    boundaryLines = np.asarray(
        [[0, 0], [1, 1.5707964e+00], [image.shape[1], 0], [image.shape[0], 1.5707964e+00]]).reshape(-1, 1, 2)
    lines = list(np.concatenate([np.asarray(lines), boundaryLines], axis=0))
    # remove redundant lines which do not  fit into the crossword pattern
    if filter:
        rho_threshold = image.shape[0] / (tot + 1)
        theta_threshold = 0.01

        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]:  # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[
                        indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now

    #print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        #print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    # draw the lines on the image and mask and save them
    mask = np.zeros_like(img)
    final_lines = []
    for line in filtered_lines:
        x1, y1, x2, y2 = calcCoordinates(line)
        tan = calcTangent(x1, y1, x2, y2)
        if tan > 1000 or tan == 0:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(mask, (x1, y1), (x2, y2), (0, 0, 255), 2)
            final_lines.append(line)

    cv2.imwrite('hough.jpg', img)
    cv2.imwrite('mask.jpg', mask)

    return final_lines, (img, mask)


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i + 1:]:
            for line1 in group:
                for line2 in next_group:
                    x11, y11, x21, y21 = calcCoordinates(line1)
                    x12, y12, x22, y22 = calcCoordinates(line2)

                    tan1 = calcTangent(x11, y11, x21, y21)
                    tan2 = calcTangent(x12, y12, x22, y22)

                    if abs(tan1 - tan2) > 0.5:
                        intersections.append(intersection(line1, line2))

    return intersections


def correctLines(lines):
    '''
    Adds some lines in case missing from standard methods
    '''

    lineDists = pd.DataFrame(np.asarray(lines).reshape(-1, 2))
    lineDists.iloc[:, 1] = (lineDists.iloc[:, 1] > 0).apply(int)
    lineDists = lineDists.sort_values(by=[1, 0])
    lineDists.columns = ['rho', 'theta']
    lineDists['rho'] = lineDists['rho'].apply(int)
    newLines = []
    for value in lineDists['theta'].unique():
        curDists = lineDists[lineDists['theta'] == value]
        curDists['delta'] = curDists['rho'] - curDists['rho'].shift(1).fillna(0)

        med = np.median(curDists['delta'].values)
        for index, dist in enumerate(curDists['delta'].values):
            if dist > 1.5 * med:
                newLines.append(
                    np.array([(curDists.iloc[index - 1, 0] + curDists.iloc[index, 0]) // 2, value]).reshape(1, 2))
    return lines + newLines


def createCrops(intersections, path = 'tmp/'):
    # cut each cell separately and place it into the tmp folder

    shutil.rmtree(path, ignore_errors=True)

    try:
        os.mkdir(path)
    except:
        pass

    xs = sorted(np.unique(np.asarray(intersections)[:, 0, 0]))
    ys = sorted(np.unique(np.asarray(intersections)[:, 0, 1]))
    for i, x in enumerate(xs[:-1]):
        for j, y in enumerate(ys[:-1]):
            cropImage = image[y:ys[j + 1], x:xs[i + 1]]
            cv2.imwrite(path + f'{j}_{i}.png', cropImage)


def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """

    image = cv2.imread(filename)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(image, lang='fra', config=r'--psm 1')  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    return text


def findArrows(data, i, j):
    '''
    Finds arrows adjacent to the current text displacement
    '''
    arrows = []
    try:
        arrow1 = data.iloc[i + 1, j]
    except:
        arrow1 = 'other'

    # if arrow is under the text fieled
    if arrow1 != 'double_text' and arrow1 != 'single_text' and arrow1 != 'other':
        if arrow1 == 'inverse_arrow':
            arrow1 = 'right'
            arrows.append([arrow1, i + 1, j])
        elif arrow1 == 'down':
            arrow1 = 'down'
            arrows.append([arrow1, i + 1, j])

    try:
        arrow2 = data.iloc[i, j + 1]
    except:
        arrow2 = 'other'

    # if arrow is to the right of the text field
    if arrow2 != 'double_text' and arrow2 != 'single_text' and arrow2 != 'other':
        if arrow2 == 'inverse_arrow':
            arrow2 = 'down'
            arrows.append([arrow2, i, j + 1])
        elif arrow2 == 'right':
            arrow2 = 'right'
            arrows.append([arrow2, i, j + 1])

    return arrows


def extractJson(data, IMG_PATH):
    json = {'definitions': []}
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data.iloc[i, j] == 'single_text' or data.iloc[i, j] == 'double_text':
                arrows = findArrows(data, i, j)
                text = ocr_core(IMG_PATH + f'{i}_{j}.png')
                for index, arrow in enumerate(arrows):
                    json['definitions'].append({
                        'label': str(text),
                        'position': [i, j],
                        'solution': {
                            'startPosition': [arrow[1], arrow[2]],
                            'direction': arrow[0]
                        }
                    })
    return json

def Im2Json(MODEL_PATH = 'model.pt', IMG_PATH = 'tmp/', IMG_SHAPE = (64, 64), CAT_MAPPING = {}):
    MATRIX_SIZE = sorted(list(map(lambda x: list(map(int, x.split('.')[0].split('_'))), os.listdir(IMG_PATH))))[-1]
    N_CLASSES = len(CAT_MAPPING)
    propertyMatrix = np.zeros((MATRIX_SIZE[0] + 1, MATRIX_SIZE[1] + 1))
    textualPropertyMatrix = pd.DataFrame(np.zeros((MATRIX_SIZE[0] + 1, MATRIX_SIZE[1] + 1)))

    # load model
    net = Net(N_CLASSES)
    net.load_state_dict(torch.load(MODEL_PATH))
    net.eval()

    # define basic image transforms for preprocessing
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

    for im in os.listdir(IMG_PATH):
        img_path = IMG_PATH + im
        image = cv2.imread(img_path)

        idxs = list(map(int, im.split('.')[0].split('_')))
        image = cv2.resize(image, IMG_SHAPE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).reshape(1, 3, IMG_SHAPE[0], IMG_SHAPE[1])
        pred = net(Variable(image)).detach().numpy()
        propertyMatrix[idxs[0], idxs[1]] = np.argmax(pred)
        textualPropertyMatrix.iloc[idxs[0], idxs[1]] = CAT_MAPPING[np.argmax(pred)]

    jsonData = extractJson(textualPropertyMatrix, IMG_PATH)
    return jsonData




if __name__ == '__main__':
    t1 = time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the crossword file")
    args = parser.parse_args()
    path = args.path

    img = cv2.imread(path)

    # find crossword lines
    print('Extracting grid from the crossword...\n\n')
    lines, (image, mask) = getLines(img, filter=True)

    # corrects some missing lines
    lines = correctLines(lines)

    # find line intersection points
    intersections = segmented_intersections(lines)

    print('Creating crop for each crossword cell...\n\n')
    # crop each cell to a separate file
    createCrops(intersections)

    print('Extracting JSON from the crossword...\n\n')
    # extract json from a given image
    jsonData = Im2Json(CAT_MAPPING=CAT_MAPPING)

    fileName = f'''{path.split('/')[-1].split('.')[0]}.json'''
    with codecs.open(fileName, 'w', encoding='utf_8_sig') as f:
        json.dump(jsonData, f, ensure_ascii=False)

    print(f'JSON file saved as {fileName} in {time() - t1} seconds!')

    import sys
    sys.exit()


