import keyboard
from pynput.mouse import Controller, Button
from pynput.keyboard import Key, Listener
import time
from dataclasses import dataclass
from PIL import Image, ImageShow, ImageOps
mouse = Controller()

brush_size = 1
image_path = "sans.png"

brush_scales = [ 0.25, 0.25 / 3, 0.25 / 5, 0.25 / 7, 0.25 / 9 ]


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

topleft = Point(855, 610)
size = Point(1538, 845)

scale = brush_scales[brush_size]

brush_points = [ Point(923, 1620), Point(1000, 1620), Point(1130, 1620), Point(1240, 1620), Point(1340, 1620) ]
# Upper left corner: 851, 610
# Bottom right corner: 2389, 1460
# Width: 1538
# Height: 1400


# Eraser: 2626, 750
# Pen: 2500, 750
# Small brush size: 923, 620

# 000000: 600, 750
# 666666: 675, 750
# 0050cd: 750, 750

# ffffff: 600, 825
# aaaaaa: 675, 825
# 26c9ff: 750, 825

# 017420: 600, 900
# 990000: 675, 900
# 964112: 750, 900

# 11b03c: 600, 990
# ff0013: 675, 990
# ff7829: 750, 990

# b0701c: 600, 1075
# 99004e: 675, 1075
# cb5a57: 750, 1075

# ffc126: 600, 1160
# ff008f: 675, 1160
# feafa8: 750, 1160

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

img = Image.open(image_path)

img_scale = min(size.x / img.width, size.y / img.height)

img_scale *= scale

scaled = ImageOps.scale(img, img_scale)

img.close()

# Dither image

color_pixels = {}

for y in range(scaled.height):
    for x in range(scaled.width):
        color = scaled.getpixel((x, y))
        palettized = get_closest_color(color, colors)
        error = (color[0] - palettized[0], color[1] - palettized[1], color[2] - palettized[2])
        
        if palettized not in color_pixels.keys():
            color_pixels[palettized] = []
        color_pixels[palettized].append(Point(x, y))

        if x < scaled.width - 1:
            scaled.putpixel((x + 1, y), add_error(scaled.getpixel((x + 1, y)), error, 7 / 16))
        if y < scaled.height - 1:
            scaled.putpixel((x, y + 1), add_error(scaled.getpixel((x, y + 1)), error, 5 / 16))
            if x < scaled.width - 1:
                scaled.putpixel((x + 1, y + 1), add_error(scaled.getpixel((x + 1, y + 1)), error, 1 / 16))
            if x > 0:
                scaled.putpixel((x - 1, y + 1), add_error(scaled.getpixel((x - 1, y + 1)), error, 3 / 16))

        scaled.putpixel((x, y), palettized)



click(50, 300)
time.sleep(0.5)

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

# Select bucket fill
click(2626, 1150)

most_color = 0
max_count = 0

for col in color_pixels.keys():
    curr_count = len(color_pixels[col])
    print(col, len(color_pixels[col]))
    if curr_count > max_count:
        max_count = curr_count
        most_color = col

clickp(color_to_point[most_color])
click(1300, 1000)

# Select brush size
clickp(brush_points[brush_size])
click(2500, 750)

# for y in range(scaled.height):
#     x = 0
#     while x < scaled.width:
#         color = scaled.getpixel((x, y))
#         point = color_to_point[color]
#         pyautogui.click(point.x, point.y)
#         pyautogui.moveTo(x / scale + topleft.x, y / scale + topleft.y)
#         pyautogui.mouseDown()

#         if x + 1 < scaled.width and scaled.getpixel((x + 1, y)) == color:
#             while x + 1 < scaled.width and scaled.getpixel((x + 1, y)) == color:
#                 x += 1
#             pyautogui.moveTo(x / scale + topleft.x, y / scale + topleft.y)

#         pyautogui.mouseUp()

#         x += 1

should_exit = False
with Listener(on_press=on_press) as listener:
    for color in color_pixels.keys():
        if color == (255, 255, 255) or color == most_color:
            continue
        point = color_to_point[color]
        clickp(point)
        points = color_pixels[color]
        i = 0
        while i < len(points):
            mouse.position = (points[i].x / scale + topleft.x, points[i].y / scale + topleft.y)
            mouse.press(Button.left)

            if i + 1 < len(points) and points[i].x == points[i + 1].x - 1 and points[i].y == points[i + 1].y:
                while i + 1 < len(points) and points[i].x == points[i + 1].x - 1 and points[i].y == points[i + 1].y:
                    i += 1
                mouse.position = (points[i].x / scale + topleft.x, points[i].y / scale + topleft.y)

            mouse.release(Button.left)
            i += 1
            time.sleep(0.03)
            if should_exit:
                break
        if should_exit:
                break
    listener.stop()
    listener.join()

ImageShow.show(scaled)