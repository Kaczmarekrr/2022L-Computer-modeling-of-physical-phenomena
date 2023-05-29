import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import os
import argparse


def make_gif():
    args = getArgs()
    path = args.input_path
    filenames = os.listdir(path)
    print(filenames[0:5])
    images = []
    for filename in tqdm(filenames):
        images.append(imageio.v2.imread(f"{path}/{filename}"))

    kargs = {"duration": args.dur,'quantizer':'nq'}
    imageio.mimsave(f"{args.input_path}.gif", images, **kargs)



def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="gif-maker")
    parser.add_argument("--input_path", type=str, help="output_directory (which is created)")
    parser.add_argument("--dur", type=float, help="duration of 1 frame")
    return parser.parse_args(argv)

if __name__ == "__main__":
    make_gif()