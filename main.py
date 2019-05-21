import argparse
import yaml
from coordinates_generator import CoordinatesGenerator
from motion_detector import MotionDetector
from colors import *
import logging
from twilio.rest import Client


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    image_file = args.image_file
    data_file = args.data_file
    start_frame = args.start_frame

    if image_file is not None:
        with open(data_file, "w+") as points:
            generator = CoordinatesGenerator(image_file, points, COLOR_RED)
            generator.generate()

    with open(data_file, "r") as data:
        points = yaml.load(data)
        detector = MotionDetector(args.video_file, points, int(start_frame))
        detector.detect_motion()


def parse_args():
    parser = argparse.ArgumentParser(description='Generates Coordinates File')

    parser.add_argument("--image",
                        dest="image_file",
                        required=False,
                        help="Image file to generate coordinates on")

    parser.add_argument("--video",
                        dest="video_file",
                        required=True,
                        help="Video file to detect motion on")

    parser.add_argument("--data",
                        dest="data_file",
                        required=True,
                        help="Data file to be used with OpenCV")

    parser.add_argument("--start-frame",
                        dest="start_frame",
                        required=False,
                        default=1,
                        help="Starting frame on the video")

    return parser.parse_args()




account_sid = 'AC0d83ee9321a5b7eaea13f951ce2ca503'
auth_token = '3e3a74fb4ab23a21c25cecdc518eb796'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="Parking Spot Detected",
                     from_='+12019925335',
                     to='+14243818357'
                 )

print(message.sid)

if __name__ == '__main__':
    main()
