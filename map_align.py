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
        cv.imwrite("img_preproc.jpg", img)
        return img

    def hough(self, img):
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, self._k3)
        # img = cv.morphologyEx(img, cv.MORPH_GRADIENT, self._k5)
        img_lines = cv.copyTo(img, None)
        # img_lines = cv.cvtColor(np.zeros(img.shape, np.uint8), cv.COLO)
        linesP = cv.HoughLinesP(img, 1, np.pi / 180, 50, None, 50, 10)
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv.LINE_AA)
        img = cv.morphologyEx(img_lines, cv.MORPH_DILATE, self._k3)
        cv.imwrite("img_hough.jpg", img_lines)
        return img_lines

    def approximate(self, mask):
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img = cv.cvtColor(np.zeros(mask.shape, np.uint8), cv.COLOR_GRAY2BGR)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > 500:
                epsilon = 0.005*cv.arcLength(cnt, True)
                approx = cv.approxPolyDP(cnt, epsilon, True)
                cv.drawContours(img, [approx], -1, (255, 255, 255), 3)
        cv.imwrite("img_approx.jpg", img)
        return img

    def sift_match(self, map, query):
        # Initiate SIFT detector
        sift = cv.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(map,None)
        kp2, des2 = sift.detectAndCompute(query,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        ratio_thresh = 0.75
        good_matches = []
        for m,n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < ratio_thresh*n.distance:
                matchesMask[i]=[1,0]
        
        draw_params = dict(matchColor = (0,255,0),
                        singlePointColor = (255,0,0),
                        matchesMask = matchesMask,
                        flags = cv.DrawMatchesFlags_DEFAULT)
        
        img = cv.drawMatchesKnn(map,kp1,query,kp2,matches,None,**draw_params)
        cv.imwrite("sift.jpg", img)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        h, mask = cv.findHomography(points1, points2, cv.RANSAC)

        # Use homography
        height, width = query.shape
        mapReg = cv.warpPerspective(map, h, (width, height))
        cv.imwrite("sift_reg.jpg", mapReg)
        return mapReg

    def surf_match(self, map, query):
        img_object = map
        img_scene = query

        #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        minHessian = 400
        detector = cv.SIFT_create()
        keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
        keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)
        #-- Step 2: Matching descriptor vectors with a FLANN based matcher
        # Since SURF is a floating-point descriptor NORM_L2 is used
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)
        #-- Filter matches using the Lowe's ratio test
        ratio_thresh = 0.75
        good_matches = []
        for m,n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        #-- Draw matches
        img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
        cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #-- Localize the object
        obj = np.empty((len(good_matches),2), dtype=np.float32)
        scene = np.empty((len(good_matches),2), dtype=np.float32)
        for i in range(len(good_matches)):
            #-- Get the keypoints from the good matches
            obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
        H, _ =  cv.findHomography(obj, scene, cv.RANSAC)
        #-- Get the corners from the image_1 ( the object to be "detected" )
        obj_corners = np.empty((4,1,2), dtype=np.float32)
        obj_corners[0,0,0] = 0
        obj_corners[0,0,1] = 0
        obj_corners[1,0,0] = img_object.shape[1]
        obj_corners[1,0,1] = 0
        obj_corners[2,0,0] = img_object.shape[1]
        obj_corners[2,0,1] = img_object.shape[0]
        obj_corners[3,0,0] = 0
        obj_corners[3,0,1] = img_object.shape[0]
        scene_corners = cv.perspectiveTransform(obj_corners, H)
        #-- Draw lines between the corners (the mapped object in the scene - image_2 )
        cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
            (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
        cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
            (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
        cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
            (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
        cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
            (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)
        #-- Show detected matches
        cv.imwrite('surf.jpg', img_matches)


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
        cv.imwrite("orb.jpg", imMatches)

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

    def get_roi(self, map, map_mask, query):
        w, h = query.shape[::-1]
        method = eval('cv.TM_CCOEFF_NORMED')

        res = cv.matchTemplate(map_mask,query,method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        roi = cv.cvtColor(map, cv.COLOR_GRAY2RGB)
        cv.rectangle(roi, top_left, bottom_right, (255, 0, 0), 2)
        cv.imwrite('roi.jpg', roi)

        mask = np.zeros(map_mask.shape[:2], dtype="uint8")
        cv.rectangle(mask, top_left, bottom_right, 255, -1)
        masked = cv.bitwise_and(map_mask, map_mask, mask=mask)
        cv.imwrite('roi_mask.jpg', masked)

        return masked

if __name__ == '__main__':

    img_path = './images/slam_map_full.png'
    map_mask_path = './images/map_mask_2.png'
    map_path = './images/map.png'
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    map_mask = cv.imread(map_mask_path, cv.IMREAD_GRAYSCALE)
    map = cv.imread(map_path, cv.IMREAD_GRAYSCALE)

    align = map_align()

    img = align.preproc(img)
    img = align.hough(img)

    roi = align.get_roi(map, map_mask, img)
    sift_match = align.sift_match(img, roi)
    surf_match = align.surf_match(img, roi)
    mapReg, h = align.registration(roi, img)