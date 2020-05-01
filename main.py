import pickle
import cv2
import helper_functions as h

dist_pickle = pickle.load(open("camera_cal/cal_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]





def process_frame(img):
    pass



def main():
    pass

if __name__ == '__main__':
    main()


