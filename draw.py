import argparse
import pickle
import time
import datetime
import math
import os
import sys
import signal
from playwright.sync_api import sync_playwright


import Gartic
from Gartic import Point

parser = argparse.ArgumentParser()
parser.add_argument(
    "--demo",
    type=str,
    help="Set up an infinite time Gartic drawing session to test out drawings",
)

args = parser.parse_args()

playwright = sync_playwright().start()
browser = playwright.firefox.launch_persistent_context(
    user_data_dir="./usr/firefox",
    headless=False,
)
page = browser.pages[0]

img: None | Gartic.Image = None
filepath = ""

if args.demo is not None:
    print("loading image")
    filepath = args.demo
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

    with open(args.demo, "rb") as file:
        img = pickle.load(file)

while True:
    if args.demo is None:
        while img is None:
            filepath = input(
                "(Enter a blank line to quit)\nEnter the path to the input .gar file: "
            )
            if filepath == "":
                print("Exiting...")
                exit()
            try:
                with open(filepath, "rb") as file:
                    img = pickle.load(file)
            except FileNotFoundError:
                print(f"File '{filepath}' not found.")

    if img is None:
        exit()

    canvas = page.locator("canvas").nth(3)
    bounds = canvas.bounding_box()

    size = Point(float(bounds["width"]), float(bounds["height"]))  # type: ignore
    print(f"Canvas size: {size.x}x{size.y}")
    print(f"Original size: {img.width}x{img.height}")
    print("Scaling by: ", img.scale_to(size.x, size.y))

    start = time.time()

    length = len(img.shapes)
    for i, shape in enumerate(img.shapes):
        shape.draw(page)

        print(f"\r{i + 1}/{length}{' ' * 10}", end="")
        if page.get_by_role("button", name="Done!").is_hidden():
            break

    print()
    Gartic.set_tool(page, Gartic.ERASER)

    page.screenshot(path=os.path.splitext(filepath)[0] + ".png")
    print("Done!")
    print(
        f"--- total time - {datetime.timedelta(seconds=math.floor(time.time() - start))} ---"
    )
    img = None

    if args.demo is not None:
        input("Press enter to quit ")
        print()
        exit()
