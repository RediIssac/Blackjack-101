# Modified code from AlexeyBY's darknet_video.py

# Authors: Hawon Park, Jeong Ho Shin, Hanbin Baik, Redi Negash

from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from suggestion import *
from sklearn.cluster import MiniBatchKMeans

# ------------------------------ BLACKJACK VARIABLES ------------------------------ #
stack = {}
players = {}
suggestions = []
allocated = []
clusters = []
acc_thresh = 75.0
num_players = 0

# ------------------------------ BLACKJACK FUNCTIONS ------------------------------ #

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

# among multiple detections, choose one most frequent ones
def choose_winner(detections):
    # dictionary for frequency
    size = len(detections[-1])

netMain = None
metaMain = None
altNames = None

# ------------------------------ YoloV3 Main Function ------------------------------ #
def YOLO():

    global metaMain, netMain, altNames
    # weightPath = "./model/50_1024_best.weights"
    weightPath = "./model/yolov3-tiny-card_best.weights"
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

    #initialization
    num_players = setup()
    clusters = []
    start = True

    # initialize webcam input (if there is an error, switch to VideoCapture(1))
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    # init for video input
    # cap = cv2.VideoCapture("./test/1player.mp4")
    # cap = cv2.VideoCapture("./test/2player.mp4")
    # cap.set(3, 1280)
    # cap.set(4, 720)
    # out = cv2.VideoWriter('2player_res.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10.0, (darknet.network_width(netMain), darknet.network_height(netMain)))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    # count the occurance of the cards
    cards_count = {}
    allocated = []
    stack = {}
    frame_count = 0
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        if frame_count % 5 == 0:
            frame_count = 1
            # detect cards
            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)

            # draw bounding boxes
            image = cvDrawBoxes(detections, frame_resized)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(1/(time.time()-prev_time))

            # compute clusters
            if(len(detections)) == 0:
                # reset()
                allocated = []
                stack = {}
                # clusters = []
                cards_count = {}
                for i in range(len(players)):
                    players[i] = []

            # 0 label, 1 accuracy, 2:0 x-coord, 2:1 y-coord, 2:2 frame width, 2:3 frame height
            for cards in detections:
                # Get the class of the detection
                temp = cards[0]
                accuracy = cards[1] * 100
                suit_num = temp.decode("utf-8")

                # if a new card is found
                if accuracy >= acc_thresh:
                    if suit_num not in stack:
                        stack[suit_num] = [cards[2][0], cards[2][1]]
                        cards_count[suit_num] = 1
                    else:
                        cards_count[suit_num] += 1

                    if len(clusters) != 0 and suit_num not in allocated and cards_count[suit_num] > 5:
                        # print("heyyyyy")
                        idx = get_closest_centroid([cards[2][0],cards[2][1]], clusters)
                        print(idx)
                        players[idx].append(suit_num)
                        allocated.append(suit_num)
                        suggestions[idx] = suggestMove(players[idx],len(allocated))
                        new_clusters = getClusters(list(stack.values()), num_players)
                        for new_cluster in new_clusters:
                            clusters[get_closest_centroid(new_cluster, clusters)] = new_cluster

            # each player has to have at least two cards
            if (start and len(stack) >= num_players*2):
                start = False
                clusters = getClusters(list(stack.values()), num_players)

        else:
            frame_count += 1

        # suggestion displaying
        for i in range(len(players)):
            if len(clusters) > 0:

                print(i, players[i], suggestions[i], clusters[i])
                temp = clusters[i]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cord_x = int(temp[0])
                cord_y = int(temp[1])

                score = getScore(players[i])
                if score > 0:
                    cv2.putText(image, suggestions[i] ,(int(clusters[i][0]), int(clusters[i][1])), font, 2,(0,0,255),2,cv2.LINE_AA)
                    cv2.putText(image, 'Score: '+str(score) ,(int(clusters[i][0])-100, int(clusters[i][1])+100), font, 2,(0,0,255),2,cv2.LINE_AA)

        cv2.imshow('Demo', image)
        # out.write(image)

        # key command (q) for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print(key)
            break

    # release video capture and close window
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    YOLO()
