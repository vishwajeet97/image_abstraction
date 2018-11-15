import os

import numpy as np
import cv2
# import scipy
import argparse

from src.toon import Toon
from src.config import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Take in the path to config file')
    parser.add_argument("--config_path", type=str, default="model/config.json",
                        help='path to the config file stored in json')
    parser.add_argument("--image_path", type=str, default="data/parrot.jpg",
                        help='path to the image')
    parser.add_argument("--result_path", type=str, default="result/parrot.jpg",
                        help='path to the store location of the abstracted image')
    
    args = parser.parse_args()
    config = Config(args.config_path)


    image = cv2.imread(args.image_path)
    toon = Toon(image, config)
    abstract_image = toon.run()

    cv2.imshow('image',image)
    cv2.imshow('abstract_image',abstract_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()