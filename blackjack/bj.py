# Modified code from AlexeyBY's Darknet

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
# from suggestion_test import *
from suggestion import *
from sklearn.cluster import MiniBatchKMeans

# ------------------------------ BLACKJACK VARIABLES ------------------------------ #
p1 = []
p2 = []
p3 = []
p4 = []
stack = []
players = {}
suggestions = []
allocated = []
clusters = []
acc_thresh = 75.0
num_players = 0

px1 = 0
px2 = 0
px3 = 0
px4 = 0

py1 = 0
py2 = 0
py3 = 0
py4 = 0

# ------------------------------ BLACKJACK FUNCTIONS ------------------------------ #
def computeHand(arr):
    val = 0
    ace = 0

    for card in arr:

        # strip the suit
        temp = card[:-1]

        # compute value for face cards
        if temp == 'J' or temp == 'K' or temp == 'Q':
            val = val + 10
        # compute value for face card: A
        elif temp == 'A':
            ace = ace + 1
        # get numeric value
        else:
            val = val + int(temp)
    return val, ace

# initalizing an array for saving centroids
centroids = []

# returns the motion movement direction in quadrant numbers
def track_frame(frame, size_back):

    #dist = sqrt(pow((x2-x1),2) + pow((y2-y1),2))

    if len(centroids) < size_back:
        return
    # print(len(centroids)-size_back)

    # we need to iterate starting from the last frame back till the size given
    back = len(centroids)-size_back
    #for i in range(back, len(centroids)):

    for c in centroids[:size_back]:
         # put text and highlight the center

        cv2.circle(frame, (c[0], c[1]), 5, (255, 255, 255), -1)

        cv2.putText(frame, "c", (c[0] - 25, c[1] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # I was trying to calculate the slope here
    x2 = centroids[len(centroids)-1][0]
    y2 = centroids[len(centroids)-1][1]
    x1 = centroids[back][0]
    y1 = centroids[back][1]


    c_y = y2-y1
    c_x = x2-x1

    # slope = (y2-y1)/(x2-x1)
    # base on the sign of the slope predicte where the direction would
    if (c_y == 0 and c_x == 0):
        return 0
    elif(c_y >=0 & c_x >=0):
        return 1
    elif(c_y <0 & c_x <0):
            return 3
    elif(c_y >=0 & c_x <= 0):
        return 2
    else:
        return 4

def process_frame(frame):
    # step 1 RGB to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = gray
    # step 2: filter video
    blur = cv2.GaussianBlur(frame, (15,15), 0)
    # find delta
    image_diff = frame - blur
    # find binary frames by thresholding
    retval, thresh = cv2.threshold(image_diff, 10, 255, cv2.THRESH_BINARY)
    frame = thresh

     # calculate moments of binary image
    M = cv2.moments(thresh)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    centroid = [cX, cY]
    centroids.append(centroid)


    return frame

# returns the centroids of each clusters
def getClusters(coords, num_players):
    kmeans = MiniBatchKMeans(n_clusters=num_players,
                                random_state=0,
                                batch_size=200).fit(coords)

    a = kmeans.cluster_centers_
    sorted(a, key=lambda k: [k[1], k[0]])
    return a

def get_closest_centroid(card, centroids):
    distances = []
    for centroid in centroids:
        euclideanDist = np.sqrt(pow((card[0]-centroid[0]),2)+pow((card[1]-centroid[1]),2))
        distances.append(euclideanDist)
    return distances.index(min(distances))

def setup():
    num_players = int(input("Number of Players? "))
    # num_players = 1
    for i in range(num_players):
        players[i] = []
        suggestions.append('')

    return num_players

# ------------------------------ YoloV3 Functions / Instance Variables ------------------------------ #
def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

netMain = None
metaMain = None
altNames = None

# ------------------------------ YoloV3 Main Function ------------------------------ #
def YOLO():

    global metaMain, netMain, altNames
    # weightPath = "./model/50_1024_best.weights"
    weightPath = "./model/yolov3-card.weights"
    configPath = "./model/yolov3-tiny-card.cfg"
    metaPath = "./model/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

    num_players = setup()
    clusters = []
    cap = cv2.VideoCapture(1)

    cap.set(3, 1280)
    cap.set(4, 720)

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        # detect cards
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

        # draw bounding boxes
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(1/(time.time()-prev_time))

        # draw boundaries for each player
        frame_height = frame_resized.shape[0]
        frame_width = frame_resized.shape[1]

        offset = 100
        if num_players == 1:
            py1 = int(100)
            px1 = int(frame_width /2) - offset
            cv2.putText(image, "Player 1",(px1, py1), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)

        elif num_players == 2:
            x = int(frame_width /2)
            y1 = 0
            y2 = frame_height
            cv2.line(image, (x,y1), (x, y2), (0,0,0), 2)

            py1 = py2 = 100
            px1 = int(frame_width / 4) - offset
            px2 = int(frame_width / 4) * 3 - offset
            cv2.putText(image, "Player 1",(px1, py1), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image, "Player 2",(px2, py2), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)

        elif num_players == 3:
            y1 = 0
            y2 = frame_height
            x1 = int(frame_width / 3)
            x2 = int(frame_width / 3) * 2
            cv2.line(image, (x1,y1), (x1, y2), (0,0,0), 2)
            cv2.line(image, (x2,y1), (x2, y2), (0,0,0), 2)

            py1 = py2 = py3 = 100
            px1 = int(frame_width / 6) - offset
            px2 = int(frame_width / 6) * 3 - offset
            px3 = int(frame_width / 6) * 5 - offset
            cv2.putText(image, "Player 1",(px1, py1), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image, "Player 2",(px2, py2), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image, "Player 3",(px3, py3), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)

        elif num_players == 4:
            cx = int(frame_width/2)
            cy = int(frame_height/2)
            cv2.line(image, (cx,0), (cx, frame_height), (0,0,0), 2)
            cv2.line(image, (0,cy), (frame_width, cy), (0,0,0), 2)

            py1 = py2 = 100
            py3 = py4 = int(frame_height / 2) + offset
            px1 = px3 = int(frame_width / 4) - offset
            px2 = px4 = int(frame_width / 4) * 3 - offset
            cv2.putText(image, "Player 1",(px1, py1), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image, "Player 2",(px2, py2), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image, "Player 3",(px3, py3), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(image, "Player 4",(px4, py4), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)

        # compute clusters for each player
        # 0 label, 1 accuracy, 2:0 x-coord, 2:1 y-coord, 2:2 frame width, 2:3 frame height
        for cards in detections:
            # Get the class of the detection
            temp = cards[0]
            accuracy = cards[1] * 100
            suit_num = temp.decode("utf-8")
            c_x = cards[2][0]
            c_y = cards[2][1]

            print(cards)

            # if a new card is found
            if accuracy >= acc_thresh and suit_num not in stack:
                if num_players == 1:
                    p1.append(suit_num)
                    stack.append(suit_num)
                # elif num_players == 2:


        if num_players == 1:
            cards_dealt = len(p2) + len(p3) + len(p4)
            move = suggestMove(p1, cards_dealt)
            print(move)
            cv2.putText(image, move,(px1, py1+100), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2,cv2.LINE_AA)




        # # suggestion displaying
        # for i in range(len(players)):
        #     print(i, players[i], suggestions[i])
        #     # print(get_closest_centroid(players[i], clusters))
        #     if len(clusters) > 0:
        #         temp = clusters[i]
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         cord_x = int(temp[0])
        #         cord_y = int(temp[1])
        #         # cv2.putText(image, suggestions[i], (x), font, 10, (255,255,255), 2, cv2.LINE_AA)
        #         cv2.putText(image, suggestions[i],(cord_x, cord_y), font, 4,(255,255,255),2,cv2.LINE_AA)

        cv2.imshow('Demo', image)
        # cv2.waitKey(3)

        # key command (q) for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(key)
            break

        # elif key == ord("f") or True:
        #     # 0 label, 1 accuracy, 2:0 x-coord, 2:1 y-coord, 2:2 frame width, 2:3 frame height
        #     for cards in detections:
        #         # Get the class of the detection
        #         temp = cards[0]
        #         accuracy = cards[1] * 100
        #         suit_num = temp.decode("utf-8")
        #
        #         # if a new card is found
        #         if accuracy >= acc_thresh:
        #             if suit_num not in stack:
        #                 stack[suit_num] = [cards[2][0], cards[2][1]]
        #
        #             if len(clusters) != 0 and suit_num not in allocated:
        #                 idx = get_closest_centroid([cards[2][0],cards[2][1]], clusters)
        #
        #                 players[idx].append(suit_num)
        #                 allocated.append(suit_num)
        #                 suggestions[idx] = suggestMove(players[idx],len(allocated))
        #
        #     if (len(stack) >= num_players*2):
        #         clusters = getClusters(list(stack.values()), num_players)

    # release video capture and close window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    YOLO()
