import argparse
import pickle
import time
import datetime
import math
from playwright.sync_api import sync_playwright


import Gartic
from Gartic import Point


parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="The .gar file path")

args = parser.parse_args()

with open(args.input, "rb") as file:
    img: Gartic.Image = pickle.load(file)


playwright = sync_playwright().start()
browser = playwright.firefox.launch_persistent_context(
    user_data_dir="./usr/firefox",
    headless=False,
)
page = browser.pages[0]
local = False
if local:
    page.goto("file:///home/flynn/Documents/GarticDrawer/Gartic%20Phone%20-%20The%20Telephone%20Game.html")
else:
    page.goto("https://garticphone.com/")
    page.get_by_role("button", name="START").click()
    page.wait_for_timeout(500)
    page.get_by_text("CUSTOM SETTINGS").click()
    page.wait_for_timeout(500)
    page.get_by_label(
        "FASTNORMALSLOWREGRESSIVEPROGRESSIVEDYNAMICINFINITEHOST'S DECISIONFASTER FIRST"
    ).select_option("6")
    page.get_by_label("WRITING, DRAWINGDRAWING,").select_option("3")
    page.wait_for_timeout(500)
    page.get_by_role("button", name="Start").click()
    page.get_by_role("button", name="Yes").click()
    page.wait_for_timeout(4000)

# input("Press enter when round has started ")

canvas = page.locator("canvas").nth(3)
bounds = canvas.bounding_box()

size = Point(float(bounds["width"]), float(bounds["height"])) # type: ignore
img_scale = size.y / img.height
print(size)
print("Scaling by", img_scale)

start = time.time()

length = len(img.shapes)
for i in range(length):
    shape = img.shapes[i] * img_scale
    shape.draw(page)

    print(f"\r{i + 1}/{length}{' ' * 10}", end="")

print()

page.screenshot(path=args.input + ".png")
print("Done!")
print(
    f"--- total time - {datetime.timedelta(seconds=math.floor(time.time() - start))} ---"
)

input("Press enter to quit ")
print()
