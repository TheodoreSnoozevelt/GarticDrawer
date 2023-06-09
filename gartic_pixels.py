import math
import subprocess
import argparse
from pynput.mouse import Controller, Button
from pynput.keyboard import Key, Listener
import time
import os
from dataclasses import dataclass
from PIL import Image, ImageShow, ImageOps, ImageFilter
import time
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='The image path', default="papy2.jpeg")
parser.add_argument('--t_pow', type=float, help='Raise the threshold value to a power', default=10)
parser.add_argument('--t_mult', type=float, help='Multiply the threshold value by a number', default=255)
parser.add_argument('--scale', type=int, help='The starting scale of the pixels', default=6)
parser.add_argument('--depth', type=int, help='The number of iterations to draw', default=4)

args = parser.parse_args()

if args.scale <= args.depth:
    args.depth = args.scale

brush_scales = [ 2**(-args.scale+n) for n in range(args.depth) ]

start_time = time.time()
mouse = Controller()

should_draw = True

def get_threshold(brush_scales, level):
    return (1 - brush_scales[level] / brush_scales[-1]) ** args.t_pow * args.t_mult

@dataclass
class Point:
    x: int
    y: int

def on_press(key):
    global should_exit
    should_exit = True

def click (x: int, y: int):
    mouse.position = (x, y)
    mouse.click(Button.left)

def clickp (pos: Point):
    click(pos.x, pos.y)

# https://stackoverflow.com/a/29643643
def to_rgb (hex: str):
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb_dist (a: tuple, b:tuple):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def get_closest_color (color: tuple, colors: list[tuple]):
    min_color = 0
    min_dist = 9999

    for col in colors:
        curr_dist = rgb_dist(color, col)
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_color = col
    
    return min_color

def add_error (color: tuple, error: tuple, factor: float):
    return (color[0] + int(error[0] * factor), color[1] + int(error[1] * factor), color[2] + int(error[2] * factor))

def apply_kernel (img: Image, size: tuple, kernel: list[float]) -> Image:
    new_img = img.copy()
    k_size = Point(int(size[0] / 2), int(size[1] / 2))

    for center_x in range(img.width):
        for center_y in range(img.height):
            color = (0, 0, 0)
            for k_index in range(size[0] * size[1]):
                kernel_value = kernel[k_index]
                k_x = k_index % size[0] - k_size.x
                k_y = math.floor(k_index / size[0]) - k_size.y
                (r,g,b) = img.getpixel(((k_x + center_x) % img.width, (k_y + center_y) % img.height))
                color = (color[0] + r * kernel_value, color[1] + g * kernel_value, color[2] + b * kernel_value)
            new_img.putpixel((center_x,center_y), color)
    
    return new_img

def dither(img: Image, grad: Image, level: int) -> dict[tuple, list[Point]]:
    color_pixels = {}

    for y in range(img.height):
        for x in range(img.width):
            color = img.getpixel((x, y))
            palettized = get_closest_color(color, colors)
            error = (color[0] - palettized[0], color[1] - palettized[1], color[2] - palettized[2])
            
            if palettized not in color_pixels.keys():
                color_pixels[palettized] = []

            if grad == None:
                color_pixels[palettized].append(Point(x, y))
            else:
                (val, _, _) = grad.getpixel((x, y))
                if len(color_pixels[palettized]) > 0:
                    p = color_pixels[palettized][-1]
                else:
                    p = Point(-1, -1)
                if val > get_threshold(brush_scales, level):
                    color_pixels[palettized].append(Point(x, y))
            

            if x < img.width - 1:
                img.putpixel((x + 1, y), add_error(img.getpixel((x + 1, y)), error, 7 / 16))
            if y < img.height - 1:
                img.putpixel((x, y + 1), add_error(img.getpixel((x, y + 1)), error, 5 / 16))
                if x < img.width - 1:
                    img.putpixel((x + 1, y + 1), add_error(img.getpixel((x + 1, y + 1)), error, 1 / 16))
                if x > 0:
                    img.putpixel((x - 1, y + 1), add_error(img.getpixel((x - 1, y + 1)), error, 3 / 16))

            img.putpixel((x, y), palettized)

    return color_pixels

def gradient(img: Image) -> Image:
    gradient_img = img.copy()

    x_grad = apply_kernel(gradient_img, (3, 3), [
        1, 0, -1, 
        2, 0, -2, 
        1, 0, -1])

    y_grad = apply_kernel(gradient_img, (3, 3), [
        1,  2,  1, 
        0,  0,  0, 
        -1, -2, -1])

    for x in range(gradient_img.width):
        for y in range(gradient_img.height):
            (xr, xg, xb) = x_grad.getpixel((x,y))
            (yr, yg, yb) = y_grad.getpixel((x,y))

            dr = math.sqrt(xr * xr + yr * yr)
            dg = math.sqrt(xg * xg + yg * yg)
            db = math.sqrt(xb * xb + yb * yb)

            length = int(math.sqrt(dr * dr + dg * dg + db * db))

            gradient_img.putpixel((x,y), (length, length, length))

    return gradient_img

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

img = Image.open(args.path)
img = img.convert("RGB")
img_scale = min(size.x / img.width, size.y / img.height)
img = ImageOps.scale(img, img_scale)

grad_path = os.path.splitext(args.path)[0] + "_grad.png"
if os.path.exists(grad_path):
    grad = Image.open(grad_path)
else:
    print("Calculating gradient...")
    grad = gradient(img)
    grad.save(grad_path)
    print("Done")

base_scaled = ImageOps.scale(img, brush_scales[0])

# Dither image
images: list[Image.Image] = []
color_pixels: list[dict[tuple, list[Point]]] = []

for i in range(len(brush_scales)):
    curr_img = ImageOps.scale(img, brush_scales[i])
    if i == 0:
        curr_grad = None
    else:
        curr_grad = ImageOps.scale(grad, brush_scales[i])
    color_pixels.append(dither(curr_img, curr_grad, i))

    if curr_grad != None:
        curr_grad.close()
    curr_img.close()

cull_count = 0
for col in colors:
    for proc_i in range(1, len(color_pixels)):
        if col not in color_pixels[proc_i]:
            continue
            
        point_i = 0
        while point_i < len(color_pixels[proc_i][col]):
            point = color_pixels[proc_i][col][point_i]

            for comp_i in range(0, proc_i):
                if col not in color_pixels[comp_i]:
                    continue

                # Calculate the scaled Point in the lower resolution image
                scl = brush_scales[comp_i] / brush_scales[proc_i]
                assert(scl < 1)
                scaled_point = Point(math.floor(point.x * scl), math.floor(point.y * scl))
                # Check if it's present in the comp_pixels using bs function
                if scaled_point in color_pixels[comp_i][col]:
                    color_pixels[proc_i][col].remove(point)
                    cull_count += 1
                    point_i -= 1
                    break

            point_i += 1

print("Removed", cull_count, "points")

if should_draw:
    click(50, 300)
    time.sleep(0.5)

    erase()

    # Select bucket fill
    click(2626, 1150)


    most_color = 0
    max_count = 0

    for col in colors:
        curr_count = 0
        for i in range(len(color_pixels)):
            if col in color_pixels[i]:
                curr_count += len(color_pixels[i][col]) * brush_scales[i]

        if curr_count > max_count:
            max_count = curr_count
            most_color = col

    clickp(color_to_point[most_color])
    click(1300, 1000)

    with Listener(on_press=on_press) as listener:
        should_exit = False

        for index in range(len(brush_scales)):
            if should_exit:
                break

            scale = brush_scales[index]
            brush_width = 1 / scale

            # Square fill
            click(2500, 1000)
            total_colors = len(color_pixels[index].keys())

            for color_i, color in enumerate(color_pixels[index].keys()):
                msg = f"Level {index+1}/{len(brush_scales)} | Color {color_i + 1}/{total_colors}"
                ps = subprocess.Popen(('echo', msg), stdout=subprocess.PIPE)
                subprocess.run(["xclip", "-i", "-selection", "clipboard"], stdin=ps.stdout)
                if should_exit:
                        break
                
                if index == 0 and color == most_color:
                    continue
                point = color_to_point[color]
                clickp(point)
                points = color_pixels[index][color]
                i = 0
                while i < len(points):
                    if should_exit:
                        break

                    mouse.position = (points[i].x * brush_width + topleft.x, points[i].y * brush_width + topleft.y)
                    mouse.press(Button.left)

                    if i + 1 < len(points) and points[i].x == points[i + 1].x - 1 and points[i].y == points[i + 1].y:
                        while i + 1 < len(points) and points[i].x == points[i + 1].x - 1 and points[i].y == points[i + 1].y:
                            i += 1
                        mouse.position = (points[i].x * brush_width + topleft.x + brush_width, points[i].y * brush_width + topleft.y + brush_width)
                    else:
                        mouse.position = (points[i].x * brush_width + topleft.x + brush_width, points[i].y * brush_width + topleft.y + brush_width)

                    mouse.release(Button.left)
                    i += 1
                    time.sleep(0.03)
    listener.stop()
    listener.join()

print("--- %s seconds ---" % (time.time() - start_time))

# ImageShow.show(scaled)