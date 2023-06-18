import argparse
from dataclasses import dataclass
import datetime
import math
import os
import pickle
import random
import select
import sys
import time
import time

import cv2
import numpy as np
from pynput.keyboard import Listener
from pynput.mouse import Button, Controller

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='The image path', default="img/papy2.jpeg")
parser.add_argument('--count', type=int, help='Number of objects to draw', default=250)
parser.add_argument('--batch', type=int, help='Number of objects to use for each batch', default=500)
parser.add_argument('--scale', type=float, help='Scale multiplier for the dimensions of the image', default=0.25)
parser.add_argument('--screenwidth', type=int, help='Horizontal resolution of your screen', default=2880)
parser.add_argument('--topstart', type=int, help='Horizontal resolution of your screen', default=245)
parser.add_argument('--blur', type=int, help='Amount to blur image when comparing', default=0)
parser.add_argument('--pickle', type=str, help="Path to pickle file of preprocessed image")
parser.add_argument('--draw', action='store_true', help="If it should draw the image to Gartic Phone after processing", default=False)
parser.add_argument('--dither', action='store_true', help="If the input image should be dithered first", default=False)

args = parser.parse_args()

og_screen_offset_y = 245
screen_offset_y = args.topstart
screen_scale = args.screenwidth / 2880

proc_scale = args.scale

start_time = time.time()
mouse = Controller()

@dataclass
class Point:
    x: int
    y: int

    def __add__(self, b):
        return Point(self.x + b.x, self.y + b.y)
    
    def __sub__(self, b):
        return self + (-b)
    
    def __neg__(self):
        return Point(-self.x, -self.y)
    
    def __mul__(self, b):
        return Point(self.x * b.x, self.y * b.y)
    
    def __truediv__(self, b):
        return Point(self.x / b.x, self.y / b.y)

@dataclass
class Rect:
    topleft: Point
    size: Point
    color: tuple[int, int, int]

def on_press(key):
    global should_exit
    if args.pickle or args.draw:
        should_exit = True

# TODO: Make these use the screen scale and offset
def transform_screen_point (x: int, y: int) -> tuple[int, int]:
    return (math.floor(x * screen_scale), math.floor((y - og_screen_offset_y) * screen_scale + screen_offset_y))

def click (x: int, y: int):
    mouse.position = transform_screen_point(x, y)
    mouse.click(Button.left)

def clickp (pos: Point):
    click(pos.x, pos.y)

def click_and_drag (a: Point, b: Point) -> None:
    mouse.position = transform_screen_point(a.y, a.x)
    mouse.press(Button.left)
    time.sleep(0.08)
    mouse.position = transform_screen_point(b.y, b.x)
    mouse.release(Button.left)

def drawrect (img: cv2.Mat, rect: Rect, thickness: int = -1) -> None:
    cv2.rectangle(img, (rect.topleft.x, rect.topleft.y, rect.size.x, rect.size.y), (rect.color[2], rect.color[1], rect.color[0]), thickness)

def imgdiff (a: cv2.Mat, b: cv2.Mat) -> float:
    if args.blur:
        b = cv2.blur(b, (args.blur, args.blur))
    diff = cv2.absdiff(a, b)
    return sum(cv2.mean(diff))

# https://stackoverflow.com/a/29643643
def to_rgb (hex: str) -> tuple[int, int, int]:
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb_dist (a: tuple[int, int, int], b:tuple[int, int, int]) -> int:
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])

def get_closest_color (color: tuple[int, int, int], colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    min_color = colors[0]
    min_dist = rgb_dist(color, colors[0])

    for i in range(1, len(colors)):
        curr_dist = rgb_dist(color, colors[i])
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_color = colors[i]
    
    return min_color

def erase() -> None:
    # Select eraser
    click(2626, 750)
    clickp(brush_points[-1])

    for y in range(0, size.y, 50):
        mouse.position = (topleft.x, y + topleft.y)
        mouse.press(Button.left)
        mouse.position = (topleft.x + size.x, y + topleft.y)
        mouse.release(Button.left)

    mouse.position = (topleft.x, topleft.y + size.y)
    mouse.press(Button.left)
    mouse.position = (topleft.x + size.x, topleft.y + size.y)
    mouse.release(Button.left)

topleft = Point(855, 610)
size = Point(1538, 845)

brush_points = [ Point(923, 1620), Point(1000, 1620), Point(1130, 1620), Point(1240, 1620), Point(1340, 1620) ]
# Upper left corner: 851, 610
# Bottom right corner: 2389, 1460
# Width: 1538
# Height: 1400


# Eraser: 2626, 750
# Pen: 2500, 750
# Square fill: 2500, 1000
# Small brush size: 923, 620

colorcodes = [ "000000", "666666", "0050cd", 
               "ffffff", "aaaaaa", "26c9ff", 
               "017420", "990000", "964112", 
               "11b03c", "ff0013", "ff7829",
               "b0701c", "99004e", "cb5a57",
               "ffc126", "ff008f", "feafa8" ]
colors = [to_rgb(c) for c in colorcodes]

x_vals = [ 600, 675, 750 ]
y_vals = [750, 825, 900, 990, 1075, 1160]
color_pos = []

for y in y_vals:
    for x in x_vals:
        color_pos.append(Point(x, y))

color_to_point = dict(zip(colors, color_pos))

def clamp (n: int, mi: int = 0, ma: int = 255) -> int:
    return min(max(n, mi), ma)

def add_error (color: tuple, error: tuple, factor: float):
    return (clamp(color[0] + int(error[0] * factor)), clamp(color[1] + int(error[1] * factor)), clamp(color[2] + int(error[2] * factor)))

def dither(img: cv2.Mat, colors: list[tuple[int, int, int]]):
    h, w = img.shape[:2]
    bgr_colors = [(x[2], x[1], x[0]) for x in colors]

    for y in range(h):
        for x in range(w):
            color = img[y, x]
            palettized = get_closest_color(color, bgr_colors)
            error = (color[0] - palettized[0], color[1] - palettized[1], color[2] - palettized[2])

            if x < w - 1:
                img[y, x + 1] = add_error(img[y, x + 1], error, 7 / 16)
            if y < h - 1:
                img[y + 1, x] = add_error(img[y + 1, x], error, 5 / 16)
                if x < w - 1:
                    img[y + 1, x + 1] = add_error(img[y + 1, x + 1], error, 1 / 16)
                if x > 0:
                    img[y + 1, x - 1] = add_error(img[y + 1, x - 1], error, 3 / 16)

            img[y, x] = palettized

def draw_gartic (final_rects: list[Rect]):
    with Listener(on_press=on_press) as listener:
        should_exit = False
        click(50, 300)
        time.sleep(0.5)

        # erase()

        # Square fill
        click(2500, 1000)

        for rect in final_rects:
            if should_exit:
                break

            clickp(color_to_point[rect.color])

            click_and_drag(rect.topleft + topleft, rect.topleft + topleft + rect.size)

            time.sleep(0.05)

        listener.stop()
        listener.join()


should_exit = False

if args.pickle:
    if not os.path.exists(args.pickle):
        print(args.pickle, "doesn't exist")
        exit()
    
    with open(args.pickle, "rb") as file:
        final_rects = pickle.load(file)
    
    draw_gartic(final_rects)

else:
    img = cv2.imread(args.path)
    img_height, img_width = img.shape[:2]
    img_scale = min(size.x / img_width, size.y / img_height) * proc_scale
    img = cv2.resize(img, (math.floor(img_width * img_scale), math.floor(img_height * img_scale)))
    img_height, img_width = img.shape[:2]

    if args.dither:
        dither(img, colors)
    if args.blur:
        img = cv2.blur(img, (args.blur, args.blur))
    
    cv2.imwrite("scaled.png", img)

    # avg = cv2.mean(img)
    # most_color = get_closest_color((avg[0], avg[1], avg[2]), colors)

    best_img = np.zeros((img_height,img_width,3), np.uint8)
    best_img[::] = (255, 255, 255)
    best_rects = [ ]
    debug_img = best_img.copy()

    def process_batch_unthreaded (original_img: cv2.Mat, evo_img: cv2.Mat, colors: list[tuple[int, int, int]]) -> tuple[cv2.Mat, None | Rect]:
        best_rect = None
        best_batch = evo_img.copy()
        best_diff = imgdiff(original_img, evo_img)
        h, w = original_img.shape[:2]
        
        for _ in range(args.batch):
            test_batch: cv2.Mat = evo_img.copy()
            test_topleft = Point(random.randint(0,w - 2), random.randint(0,h - 2))
            test_size = Point(random.randint(1, w - test_topleft.x), random.randint(1, h - test_topleft.y))
            roi = img[test_topleft.y:(test_topleft.y+test_size.y), test_topleft.x:(test_topleft.x+test_size.x)]
            avg_col = cv2.mean(roi)
            test_color = get_closest_color((avg_col[2], avg_col[1], avg_col[0]), colors)

            test_rect = Rect(test_topleft, test_size, test_color)
            drawrect(test_batch, test_rect)

            test_diff = imgdiff(original_img, test_batch)
            if test_diff < best_diff:
                best_batch = test_batch
                best_diff = test_diff
                best_rect = test_rect

        return (best_batch, best_rect)

    avg_step_time = 0
    click(50, 300)
    time.sleep(0.5)

    # erase()

    # Square fill
    click(2500, 1000)
    with Listener(on_press=on_press) as listener:
        for j in range(args.count):
            if should_exit:
                break

            avg_step_time = (time.time() - start_time) / (j + 1)
            
            if j == (args.count / 10):
                print(f"\rEstimated total time - {datetime.timedelta(seconds=math.floor(avg_step_time * args.count))}                              ")

            time_left = avg_step_time * (args.count - j)
            print(f"\r{j + 1}/{args.count} Estimated time left - {datetime.timedelta(seconds=math.floor(time_left))}             ", end="")

            best_batch, best_rect = process_batch_unthreaded(img, best_img, colors)

            if best_rect != None:
                best_img = best_batch.copy()
                scl_tl = best_rect.topleft / Point(proc_scale, proc_scale)
                scl_s = best_rect.size / Point(proc_scale, proc_scale)
                scaled_rect = Rect(scl_tl, scl_s, best_rect.color)
                # drawrect(debug_img, best_rect, 1)
                best_rects.append(scaled_rect)
                clickp(color_to_point[scaled_rect.color])

                if args.draw:
                    click_and_drag(scaled_rect.topleft + topleft, scaled_rect.topleft + topleft + scaled_rect.size)

                    time.sleep(0.02)
            
            if (j + 1) % 100 == 0:
                cv2.imwrite("evolution.png", best_img)
                cv2.imwrite("debug.png", debug_img)
            
            if select.select([sys.stdin, ], [], [], 0.0)[0]:
                should_exit = True

        listener.stop()
        listener.join()
    
    print()

    cv2.imwrite("evolution.png", best_img)
    # cv2.imwrite("debug.png", debug_img)

    with open("rects.pickle", "wb") as file:
        pickle.dump(best_rects, file)
    
print(f"--- total time - {datetime.timedelta(seconds=math.floor(time.time() - start_time))} ---")
