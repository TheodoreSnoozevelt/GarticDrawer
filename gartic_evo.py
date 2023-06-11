import math
import random
import pickle
import argparse
import datetime
import sys
import select
from pynput.mouse import Controller, Button
from pynput.keyboard import Key, Listener
import time
import os
from dataclasses import dataclass
import cv2
import time
import numpy as np
import threading
from queue import Queue

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='The image path', default="img/papy2.jpeg")
parser.add_argument('--count', type=int, help='Number of objects to draw', default=100)
parser.add_argument('--batch', type=int, help='Number of objects to use for each batch', default=5000)
parser.add_argument('--thread', type=int, help="Number of threads to use to process batches", default=4)
parser.add_argument('--pickle', type=str, help="Path to pickle file of preprocessed image")

proc_scale = 0.15

args = parser.parse_args()

start_time = time.time()
mouse = Controller()

@dataclass
class Point:
    x: int
    y: int

@dataclass
class Rect:
    topleft: Point
    size: Point
    color: tuple[int, int, int]

def on_press(key):
    global should_exit
    should_exit = True

def click (x: int, y: int):
    mouse.position = (x, y)
    mouse.click(Button.left)

def clickp (pos: Point):
    click(pos.x, pos.y)

def drawrect (img: cv2.Mat, rect: Rect, thickness: int = -1) -> None:
    cv2.rectangle(img, (rect.topleft.x, rect.topleft.y, rect.size.x, rect.size.y), rect.color, thickness)

def imgdiff (a: cv2.Mat, b: cv2.Mat) -> float:
    diff = cv2.absdiff(a, b)
    return sum(cv2.mean(diff))

# https://stackoverflow.com/a/29643643
def to_rgb (hex: str) -> tuple[int, int, int]:
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb_dist (a: tuple[int, int, int], b:tuple[int, int, int]) -> int:
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2])

def get_closest_color (color: tuple[int, int, int], colors: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    min_color = 0
    min_dist = 9999

    for col in colors:
        curr_dist = rgb_dist(color, col)
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_color = col
    
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
should_exit = False

if args.pickle:
    if not os.path.exists(args.pickle):
        print(args.pickle, "doesn't exist")
        exit()
    
    with open(args.pickle, "rb") as file:
        final_rects = pickle.load(file)
    
    click(50, 300)
    time.sleep(0.5)

    erase()

    with Listener(on_press=on_press) as listener:
        # Square fill
        click(2500, 1000)

        for rect in final_rects:
            if should_exit:
                break

            clickp(color_to_point[rect.color])

            mouse.position = ((rect.topleft.x, rect.topleft.y))
            mouse.press(Button.left)
            mouse.position = ((rect.topleft.x + rect.size.x, rect.topleft.y + rect.size.y))
            mouse.release(Button.left)

            time.sleep(0.03)
    listener.stop()
    listener.join()

else:
    img = cv2.imread(args.path)
    img_height, img_width = img.shape[:2]
    img_scale = min(size.x / img_width, size.y / img_height) * proc_scale
    img = cv2.resize(img, (math.floor(img_width * img_scale), math.floor(img_height * img_scale)))
    img_height, img_width = img.shape[:2]

    avg = cv2.mean(img)
    most_color = get_closest_color((avg[0], avg[1], avg[2]), colors)

    best_img = np.zeros((img_height,img_width,3), np.uint8)
    best_img[::] = most_color
    best_rects = [ Rect(Point(0, 0), Point(img_width, img_height), most_color) ]

    # https://stackoverflow.com/a/25904681
    class BatchThread(threading.Thread):
        def __init__(self, queue: Queue, lock: threading.Lock, img: cv2.Mat, batch_size: int, colors: list[tuple[int, int, int]], return_vals: list[tuple[cv2.Mat, Rect, float]], index: int, args=(), kwargs=None):
            threading.Thread.__init__(self, args=(), kwargs=None)
            self.queue: Queue[cv2.Mat] = queue
            self.lock: threading.Lock = lock
            self.img: cv2.Mat = img
            self.batch_size: int = batch_size
            self.colors: list[tuple[int, int, int]] = colors
            self.return_vals: list[tuple[cv2.Mat, Rect, float]] = return_vals
            self.index: int = index
            self.daemon = True
            # self.receive_messages = args[0]

        def run(self):
            while True:
                val: cv2.Mat = self.queue.get()
                if val is None:   # If you send `None`, the thread will exit.
                    return
                self.process_batch(val)

        def process_batch(self, evo_img: cv2.Mat):
            with self.lock:
                best_rect: Rect = None
                best_batch: cv2.Mat = evo_img.copy()
                best_diff = imgdiff(self.img, evo_img)
                h, w = self.img.shape[:2]
                
                for _ in range(self.batch_size):
                    test_batch: cv2.Mat = evo_img.copy()
                    test_topleft = Point(random.randint(0,w - 2), random.randint(0,h - 2))
                    test_size = Point(random.randint(1,w - test_topleft.x - 1), random.randint(1,h - test_topleft.y - 1))
                    avg_col = cv2.mean(self.img[test_topleft.x:test_size.x, test_topleft.y:test_size.y])
                    # test_color = (avg_col[0], avg_col[1], avg_col[2])
                    test_color = get_closest_color((avg_col[0], avg_col[1], avg_col[2]), self.colors)
                    test_rect = Rect(test_topleft, test_size, test_color)
                    drawrect(test_batch, test_rect)

                    test_diff = imgdiff(self.img, test_batch)
                    if test_diff < best_diff:
                        best_batch = test_batch
                        best_diff = test_diff
                        best_rect = test_rect
                
                self.return_vals[self.index] = (best_batch, best_rect, best_diff)
                print(f"Thread {self.index} done")

    def process_batch_threaded (evo_img: cv2.Mat) -> tuple[cv2.Mat, Rect]:
        for t_index in range(args.thread):
            thread_queues[t_index].put(evo_img)

        best_batch, best_rect, best_diff = (None, None, 99999999999999999999999999)

        processed_all = True

        while processed_all:
            for i in range(args.thread):
                if return_vals[i] == None or return_vals[i] == -1:
                    continue

                temp_batch, temp_rect, temp_diff = return_vals[i]
                
                return_vals[i] = -1
                processed_all = False
                if temp_diff < best_diff:
                    best_batch = temp_batch
                    best_rect = temp_rect
                    best_diff = temp_diff
        
        for i in range(args.thread):
            with return_vals_locks[i]:
                return_vals[i] = None
        
        return (best_batch, best_rect)
    
    def process_batch_unthreaded (img: cv2.Mat, evo_img: cv2.Mat, colors: list[tuple[int, int, int]]) -> tuple[cv2.Mat, Rect]:
        best_rect: Rect = None
        best_batch: cv2.Mat = evo_img.copy()
        best_diff = imgdiff(img, evo_img)
        h, w = img.shape[:2]
        
        for i in range(args.batch):
            test_batch: cv2.Mat = evo_img.copy()
            test_topleft = Point(random.randint(0,w - 2), random.randint(0,h - 2))
            test_size = Point(random.randint(1,w - test_topleft.x - 1), random.randint(1,h - test_topleft.y - 1))
            avg_col = cv2.mean(img[test_topleft.x:test_size.x, test_topleft.y:test_size.y])
            # test_color = (avg_col[0], avg_col[1], avg_col[2])
            test_color = get_closest_color((avg_col[0], avg_col[1], avg_col[2]), colors)
            test_rect = Rect(test_topleft, test_size, test_color)
            drawrect(test_batch, test_rect)

            test_diff = imgdiff(img, test_batch)
            if test_diff < best_diff:
                best_batch = test_batch
                best_diff = test_diff
                best_rect = test_rect
        
        return (best_batch, best_rect)

    threads = [None] * args.thread
    return_vals = [None] * args.thread
    return_vals_locks = [None] * args.thread
    thread_queues = [None] * args.thread

    for thread_index in range(args.thread):
            return_vals_locks[thread_index] = threading.Lock()
            thread_queues[thread_index] = Queue()
            thread = BatchThread(thread_queues[thread_index], return_vals_locks[thread_index], img, math.floor(args.batch / args.thread), colors, return_vals, thread_index)
            threads[thread_index] = thread
            thread.start()

    avg_step_time = 0
    last_step = time.time()

    for j in range(args.count):
        if should_exit:
            break

        if j < 200:
            avg_step_time *= j
            avg_step_time += time.time() - last_step
            avg_step_time /= j + 1
            last_step = time.time()

        time_left = avg_step_time * (args.count - j)
        print(f"\r{j}/{args.count} estimated time left - {datetime.timedelta(seconds=math.floor(time_left))}             ", end="")

        if args.thread > 1:
            best_batch, best_rect = process_batch_threaded(best_img)
        else:
            best_batch, best_rect = process_batch_unthreaded(img, best_img, colors)
        if best_rect != None:
            best_img = best_batch.copy()
            scl_tl = Point(best_rect.topleft.x / proc_scale, best_rect.topleft.y / proc_scale)
            scl_s = Point(best_rect.size.x / proc_scale, best_rect.size.y / proc_scale)
            scaled_rect = Rect(scl_tl, scl_s, best_rect.color)
            best_rects.append(scaled_rect)
        
        if select.select([sys.stdin, ], [], [], 0.0)[0]:
            should_exit = True

    print("Closing threads...")
    print()
    for i in range(args.thread):
        thread_queues[i].put(None)
        print(f"\r{i + 1}/{args.thread} ", end="")
        threads[i].join()
    print()
    print("Done")

    print()

    cv2.imwrite("scaled.png", img)
    cv2.imwrite("evolution.png", best_img)

    with open("rects.pickle", "wb") as file:
        pickle.dump(best_rects, file)

print(f"--- total time - {datetime.timedelta(seconds=math.floor(time.time() - start_time))} ---")

# ImageShow.show(scaled)