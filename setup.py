import argparse
from playwright.sync_api import sync_playwright


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--manual", action="store_true", help="Do nothing automatically and let the user set things up")
args = parser.parse_args()

playwright = sync_playwright().start()
browser = playwright.firefox.launch_persistent_context(
    user_data_dir="./usr/firefox",
    headless=False,
)
page = browser.pages[0]

if args.manual:
    print("Running in manual mode. Press enter to quit.")
    input()
    print()
    exit()

page.goto("https://addons.mozilla.org/en-US/firefox/addon/ublock-origin/")
print("Please install this extension.")
# page.get_by_role("link", name="Add to Firefox").click()

print("Accept the usage conditions and press enter when done to close the program.")
input()
print()
