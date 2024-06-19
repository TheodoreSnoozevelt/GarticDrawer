import argparse
import pickle
import os
import tempfile
import subprocess
import shutil
import signal

from cv2.typing import MatLike
import numpy as np
import cv2
from pygifsicle import optimize

import Gartic
from Gartic import Image, Point
from OpencvDrawing import draw_shape

written_imgs: list[str] = []

def writeimg(i: int, output_img: MatLike) -> None:
    path = os.path.join(tmp, f"{i:05d}.png")
    cv2.imwrite(path, output_img)
    written_imgs.append(path)


def cleanup() -> None:
    shutil.rmtree(tmp)


def shutdown() -> None:
    cleanup()
    exit(2)


signal.signal(signal.SIGINT, lambda _, b: shutdown())

parser = argparse.ArgumentParser()
parser.add_argument(
    "input",
    type=str,
    help="Path to the input .gar file",
)
parser.add_argument(
    "--height",
    type=int,
    help="Vertical resolution of output GIF",
    default=1080,
)
parser.add_argument(
    "--frames",
    type=int,
    help="Number of frames to add to the GIF",
    default=100,
)

args = parser.parse_args()

with open(args.input, "rb") as file:
    img: Image = pickle.load(file)

size = Point(img.width * args.height / img.height, args.height)
img.scale_to(size.x, size.y)
thicknessScale = img.height / 400

if len(img.shapes) < args.frames:
    args.frame = len(img.shapes)

drawn_img = np.zeros((int(size.y), int(size.x), 3), np.uint8)
drawn_img[::] = Gartic.colors[img.shapes[0].colorIndex]

length = len(img.shapes)
filepath = os.path.splitext(args.input)[0] + ".gif"
tmp = os.path.join(tempfile.gettempdir(), "make-gif")
if os.path.exists(tmp):
    cleanup()
os.mkdir(tmp)

img_interval = length / args.frames
imgs = 0
for i, shape in enumerate(img.shapes):
    if i == 0:
        writeimg(i, drawn_img)
        imgs += 1
        continue

    draw_shape(drawn_img, shape, thicknessScale)

    if i >= imgs * img_interval or i == length - 1:
        writeimg(i, drawn_img)
        imgs += 1

    print(f"\r{i + 1}/{length}{' ' * 10}", end="")

for i in range(length, length + 40):
    writeimg(i, drawn_img)
print()
print("Converting file...")
ret = subprocess.run(
    [
        "gifski",
        "-o",
        filepath,
        *written_imgs,
    ]
)
if ret.returncode != 0:
    print("Something went wrong converting images to a GIF")
    cleanup()
    exit(1)
cleanup()
