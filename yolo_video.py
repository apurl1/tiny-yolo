import sys
import argparse
import pandas as pd
import numpy as np
from yolo import YOLO, detect_video
from PIL import Image

def detect_img(yolo):
    results = []
    i = 0
    while True:
        #img = input('Input image filename:')
        img = 'data/images/Trial1_Right-051.jpeg'
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
        i += 1
    yolo.close_session()
    return results

def parse(data, filepath):
    results = {'Label': [], 'Confidence': [], 'Box': []}
    if (len(data) == 0):
        print("No objects detected")
        return
    data = np.array(data)
    labels = data[:, 0]
    scores = data[:, 1]
    boxes = data[:, 2]
    results['Label'] = labels
    results['Confidence'] = scores
    results['Box'] = boxes
    df = pd.DataFrame(results)
    df.to_csv(filepath)

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    parser.add_argument(
        "--csv", nargs='?', type=str, required = False, default="",
        help = "CSV file path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        results = detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        print(FLAGS.csv)
        if FLAGS.csv != "":
            results = detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
            parse(results, FLAGS.csv)
        else:
            detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
