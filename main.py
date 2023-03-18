import numpy as np
import matplotlib.pyplot as plt
import cv2

def display(img, cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    plt.imshow(img, cmap=cmap)
    plt.show()

"""
using existing skills, 
finds all the coins but as a big blob
"""

sep_coins = cv2.imread('../DATA/pennies.jpg')
# display(sep_coins)

# Median Blur
# sep_blur = cv2.medianBlur(sep_coins, 25)
# display(sep_blur)

# Grayscale
# gray_sep_coins = cv2.cvtColor(sep_blur, cv2.COLOR_BGR2GRAY)
# display(gray_sep_coins)

# Binary Threshold
# ret, sep_thresh = cv2.threshold(gray_sep_coins, 160, 255, cv2.THRESH_BINARY_INV)
# display(sep_thresh)

# Find Contours
# contours, hierarchy = cv2.findContours(sep_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
# for i in range(len(contours)):
#     if hierarchy[0][i][3] == -1:
#         cv2.drawContours(sep_coins,contours, i,(233,0,0), 10)
#
# display(sep_coins)

"""
using watershed algorithm
"""

img = cv2.imread('../DATA/pennies.jpg')

# blur image
img = cv2.medianBlur(img, 35)

# gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# more advanced thresholding using otsu's threshold algorithm
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# get the background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# distance transform to find peaks
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# these points are absolutely certain to be in the foreground
ret, sure_fg = cv2.threshold(dist_transform, .7*dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1 #sets the color of the definitely foreground items (the donut hole aka SEEDS)
markers[unknown==255] = 0 #sets the color of the items that may or may not be foreground/background (the donuts)
markers = cv2.watershed(img, markers) #gets unique sections based on seeds
display(markers)

# Find Contours
contours, hierarchy = cv2.findContours(markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins, contours, i, (233,0,0), 10)

display(sep_coins)