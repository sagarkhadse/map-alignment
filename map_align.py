import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.05

class map_align:

    _k3 = np.ones((3, 3), np.uint8)
    _k5 = np.ones((5, 5), np.uint8)

    def __init__(self) -> None:
        pass
    
    def edges(self, img):
        _, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
        img = cv.morphologyEx(img, cv.MORPH_DILATE, self._k5)
        img = cv.Canny(img, 100, 200)
        cv.imwrite("edges.jpg", img)
        return img

    def preproc(self, img):
        _, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
        img = cv.morphologyEx(img, cv.MORPH_DILATE, self._k5)
        # img = cv.morphologyEx(img, cv.MORPH_GRADIENT, self._k5)

        return img

        img_lines = cv.cvtColor(np.zeros(img.shape, np.uint8), cv.COLOR_GRAY2BGR)
            
        linesP = cv.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 1, cv.LINE_AA)
        
        return img_lines

    def approximate(self, mask):
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img = cv.cvtColor(np.zeros(mask.shape, np.uint8), cv.COLOR_GRAY2BGR)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 500:
                epsilon = 0.001*cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
                cv.drawContours(img, [approx], -1, (255, 255, 255), 3)
        cv.imwrite("approx.jpg", img)
        return img

    def sift_match(self, map, query):
        # Initiate SIFT detector
        sift = cv.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(map,None)
        kp2, des2 = sift.detectAndCompute(query,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_DEFAULT)
        img = cv.drawMatchesKnn(map,kp1,query,kp2,matches,None,**draw_params)

        return img

    def registration(self, map, query):

        # Detect ORB features and compute descriptors.
        orb = cv.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(map, None)
        keypoints2, descriptors2 = orb.detectAndCompute(query, None)

        # Match features.
        matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = list(matcher.match(descriptors1, descriptors2, None))

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # Draw top matches
        imMatches = cv.drawMatches(map, keypoints1, query, keypoints2, matches, None)
        cv.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv.findHomography(points1, points2, cv.RANSAC)

        # Use homography
        height, width = query.shape
        mapReg = cv.warpPerspective(map, h, (width, height))

        return mapReg, h

    def get_roi(self, map, query):
        w, h = query.shape[::-1]
        method = eval('cv.TM_CCOEFF_NORMED')

        res = cv.matchTemplate(map,query,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        roi = cv.cvtColor(map, cv.COLOR_GRAY2RGB)
        cv.rectangle(roi, top_left, bottom_right, (255, 0, 0), 2)
        cv.imwrite('roi.jpg', roi)
        return map[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

if __name__ == '__main__':

    img_path = './images/slam_map_full.png'
    map_mask_path = './images/map_mask_2.png'
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    map_mask = cv.imread(map_mask_path, cv.IMREAD_GRAYSCALE)

    align = map_align()
    edges = align.edges(img)
    img_lines = align.preproc(img)
    approx = align.approximate(img_lines)
    
    
    roi = align.get_roi(map_mask, img_lines)
    sift_img = align.sift_match(roi, img_lines)
    mapReg, h = align.registration(img_lines, roi);
    print(h)

    plt.imshow(mapReg, 'gray')
    plt.show()