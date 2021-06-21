import cv2
import math
import os 
import argparse

get_image_name = lambda count, h, w : "{}x{}-".format(w, h) + ("0"*15 + str(count))[-14:]
OUTPUT_DIR = "./drive/data/images/flxR/"

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="video splitter")
    parser.add_argument(
        "-v", "--video", help="path to the video to split",
         required=True)
    
    parser.add_argument(
        "-f", "--fps", help="preferred FPS",
        default=None, type=float)

    parser.add_argument(
        "-s", "--sec", help="preferred start second",
        default=0, type=float)

    parser.add_argument(
        "-o", "--output", help="output path",
        default=OUTPUT_DIR)

    parser.add_argument(
        "-d", "--details", help="output path",
        default=None)
    
    return parser.parse_args()

def getVideoDetails(video_file):
    video = cv2.VideoCapture(video_file)
    details = {
        "fps" : video.get(cv2.CAP_PROP_FPS),
        "length" : "length of video to do"
    }
    return details

def processVideo(video_file, out_dir, sec=0.0, preferred_fps=None):
    vidcap = cv2.VideoCapture(video_file)
    fps = preferred_fps if preferred_fps else vidcap.get(cv2.CAP_PROP_FPS)
    step = 1/fps

    images = []
    success = True
    
    count = 0
    height, width = [None, None]
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        success,image = vidcap.read() 

        if not success:
            break

        if not height:
            height, width, channels = image.shape

        sec += step
        # get name and write to disk 
        path = "{}/{}.png".format(out_dir, get_image_name(count, height, width))
        cv2.imwrite(path, image)

        count += 1

    vidcap.release()

    return count


if __name__ == '__main__':
    args = parse_args()
    if args.details : # user want details about video
        details = getVideoDetails(args.video)
        print (details)
        exit(0)
    
    sec = float(args.sec)

    preferred_fps = args.fps if args.fps else None

    out_dir = args.output.strip()

    if out_dir.endswith("/"):
        out_dir = out_dir = out_dir[:-1]
    
    images_count = processVideo(args.video, out_dir, preferred_fps = preferred_fps, sec=sec)

    print("finished! written to disk {} images in {}...".format(images_count, out_dir))