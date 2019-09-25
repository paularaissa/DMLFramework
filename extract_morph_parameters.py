import glob
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi  # to determine shape centrality
from skimage.measure import label
from skimage.measure import regionprops
from skimage.morphology import convex_hull_image
from skimage.util import invert
import fractal_box_count
import pandas as pd

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')
    plt.show()

def calculateDistance(x1,x2):
    try:
        y1 = x1[0][1]
    except:
        y1 = x1[1]
    try:
        x1_1 = x1[0][0]
    except:
        x1_1 = x1[0]
    y2 = x2[0][1]
    x2_2 = x2[0][0]
    dist = math.sqrt((x2_2 - x1_1)**2 + (y2 - y1)**2)
    return dist


path = "/home/paula/Doutoramento/imagesTiff/TIFF"
OUTPUT_FILE = "/home/paula/Doutoramento/imagesTiff/morphological_results.csv"

all_paths = []
for root, dirs, files in os.walk(path):
    all_paths.append(dirs)

df = pd.DataFrame()

for dir in all_paths[0]:
    files = glob.glob(path + "/" + dir + "/*.tif")
    images = []
    for imagepath in files:
        img = cv2.imread(str(imagepath))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        blur = cv2.blur(gray, (3, 3))  # blur the image
        ret, thresh = cv2.threshold(blur, 5, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        inv = invert(thresh)
        morph_kernel = np.ones((15, 15), np.uint8)
        cells_morph = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, morph_kernel)

        # find binary image with edges
        cells_edges = cv2.Canny(thresh, threshold1=90, threshold2=110)

        # find contours
        contours, hierarchy = cv2.findContours(invert(cells_morph), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Find object with the biggest bounding box
        mx = (0, 0, 0, 0)  # biggest bounding box so far
        mx_area = 0
        for cont in contours:
            x, y, w, h = cv2.boundingRect(cont)
            area = w * h
            if area > mx_area:
                mx = x, y, w, h
                mx_area = area

        x, y, w, h = mx
        roi = cells_morph[y:y + h, x:x + w]  # cutted cell
        cv2.rectangle(inv, (x, y), (x + w, y + h), (200, 0, 0), 2)
        contour_cell, hierarchy_cell = cv2.findContours(invert(roi), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Convex Hull
        chull = convex_hull_image(invert(roi))
        # using image processing module of scipy to find the center of the convex hull
        cy, cx = ndi.center_of_mass(chull)
        # Find Countours
        #contours = measure.find_contours(invert(roi), .8)
        #contour = max(contours, key=len)
        labels2 = label(chull, background=0)
        for region in regionprops(labels2):
            CHA_area = region.area  # convex hull area
            CHA_perimeter = region.perimeter  # convex hull perimeter

        # Cell area and Cell perimeter
        label_area = label(invert(roi), background=0)
        for region in regionprops(label_area):
            cell_area = region.area
            cell_perimeter = region.perimeter

        # Density
        density = cell_area / CHA_area

        # Roughness
        roughness = cell_perimeter / CHA_perimeter

        # Convex Hull Circularity
        chc = (4 * math.pi * cell_area) / (cell_perimeter)

        # Circularity
        cc = (4 * math.pi * CHA_area) / (CHA_perimeter)

        cnt = contour_cell[0]
        M = cv2.moments(cnt)

        # Centroid
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        # Perimetro do contorno
        perimeter = cv2.arcLength(cnt,True)

        # Area do contorno
        area = cv2.contourArea(cnt)

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = [x, y]

        # Diameter of the bounding circle
        diameter = int(radius) * 2

        # The mean radius
        mean_radius = radius

        #Convex hull span ratio (form factor)
        form_factor = (4 * math.pi * CHA_area) / CHA_perimeter ** 2

        hull = cv2.convexHull(cnt, returnPoints=True)
        lower = []
        max_hull_radii = []
        for i in range(len(hull)-1):
            p = calculateDistance(hull[i], hull[i+1])
            hull_radii = calculateDistance(center, hull[i])
            lower.append(p)
            max_hull_radii.append(hull_radii)
        maximum_hull_radii = max(max_hull_radii)

        # The ratio maximum/minimum convex hull radii
        ratio_CHA_radii = maximum_hull_radii / radius

        # Maximum span across the convex hull
        maximum_distance = max(lower)

        # Fractal box count
        coeffs = fractal_box_count.fractal_dimension(roi)
        fractal = -coeffs[0]
        lacunarity = coeffs[1]

        data = [fractal, lacunarity, cell_area, CHA_area, density, cell_perimeter, CHA_perimeter, roughness, form_factor, cc, chc, diameter, maximum_distance, maximum_hull_radii, mean_radius]

        temp = pd.DataFrame({"Path":[os.path.basename(os.path.dirname(imagepath))],"Image":[os.path.basename(imagepath)],"D":[fractal], "A":[lacunarity], "CellArea":[cell_area], "CHA":[CHA_area], "Density":[density], "CellPerimeter":[cell_perimeter],
             "CHP":[CHA_perimeter], "Roughness":[roughness], "SpRatio":[form_factor], "CC":[cc], "CHC":[chc], "Diameter":[diameter], "MSACH":[maximum_distance],
             "HullRadii": [maximum_hull_radii], "MeanRadius":[mean_radius]})
        df = pd.concat([df, temp])

df.to_csv(OUTPUT_FILE, sep='\t', encoding='utf-8')