import math
import os
from enum import Enum
from pathlib import Path
import requests
from bs4 import BeautifulSoup

from PIL import Image, ImageDraw
from skimage.measure import compare_ssim
import logging
import numpy as np
import pyvips
import glob
import json
import subprocess
from datetime import datetime
import re
import zipfile

CACHES_BASE_URL = "https://archive.openrs2.org"

LOG = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = 1000000000000000

TILE_SIZE_PX = 256

MIN_ZOOM = 3
MAX_ZOOM = 11

MIN_Z = 0
MAX_Z = 3

REPO_DIR = '/repo' # Name of the directory mounted on the local machine
OUTPUT_DIR = os.path.join(REPO_DIR, 'output/')
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache/')
DIFF_DIR = os.path.join(OUTPUT_DIR, 'diff/')
XTEA_FILE = os.path.join(CACHE_DIR, "xteas.json")
GENERATED_FULL_IMAGES = os.path.join(OUTPUT_DIR, 'generated_images/')
TILE_DIR = REPO_DIR

image_prefix = "full_image_"


logging.basicConfig(
    format='%(asctime)s %(levelname)-4s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)


class Side(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4


def main():
    os.makedirs(DIFF_DIR, exist_ok=True)

    LOG.info("Downloading cache & XTEAs")
    download_cache()

    LOG.info("Building map base images")
    build_full_map_images()

    for plane in range(MAX_Z + 1):
        LOG.info(f"Generating plane {plane}")
        LOG.info("Loading images into memory")
    
        old_image_location = os.path.join(REPO_DIR, f"full_image_{plane}.png")
        new_image_location = os.path.join(GENERATED_FULL_IMAGES, f"img-{plane}.png")

        old_image = pyvips.Image.new_from_file(old_image_location)
    
        new_image = pyvips.Image.new_from_file(new_image_location)
    
        image_width = new_image.width
        image_width_tiles = int(image_width / TILE_SIZE_PX)
    
        starting_zoom = int(math.sqrt(math.pow(2, math.ceil(math.log(image_width_tiles) / math.log(2)))))
    
        LOG.info("Calculating changed tiles")
        changed_tiles = get_changed_tiles(old_image, new_image, starting_zoom)
    
        LOG.info("Storing diff image")
        output_tile_diff_image(changed_tiles, new_image_location,  str(Path(DIFF_DIR, f"diff_{plane}.png")))
    
        LOG.info(f"Found {len(changed_tiles)} changed tiles at zoom level {starting_zoom}")
    
        LOG.info(f"Saving changed tiles at zoom level {starting_zoom}")
        for tile in changed_tiles:
            save_tile(tile["image"], plane, starting_zoom, tile["x"], tile["y"])
    
        next_changed_tiles = changed_tiles
        for zoom in range(starting_zoom + 1, MAX_ZOOM + 1):
            LOG.info(f"Splitting changed tiles from zoom level {zoom-1} to zoom level {zoom}")
            next_changed_tiles = split_tiles_to_new_zoom(next_changed_tiles, plane, zoom)
            LOG.info("Done")
    
        for zoom in reversed(range(MIN_ZOOM + 1, starting_zoom + 1)):
            LOG.info(f"Joining changed tiles from zoom level {zoom} to zoom level {zoom - 1}")
            changed_tiles = join_tiles_to_new_zoom(changed_tiles, plane, zoom, zoom - 1)
            LOG.info("Done")

        subprocess.run(['cp', new_image_location, old_image_location], check=True)


def download_cache():
    caches = requests.get(CACHES_BASE_URL + "/caches", allow_redirects=True)
    soup = BeautifulSoup(caches.content)
    cache_table = soup.find('table')
    table_body = cache_table.find('tbody')
    rows = table_body.find_all('tr')

    header = cache_table.find("thead")
    header_cols = header.find_all('th')
    header_texts = [ col.text.strip().lower() for col in header_cols ]
    game_index = header_texts.index("game")
    timestamp_index = header_texts.index("timestamp")
    links_index = header_texts.index("links")

    oldschool_rows = []
    for row in rows:
        columns = row.find_all('td')
        if columns[game_index].text.strip().lower() == "oldschool":
            oldschool_rows.append(row.find_all('td'))

    max_old_school_row = None
    max_old_school_timestamp = None

    for row in oldschool_rows:
        timestamp_str = re.sub(r'\s+', '', row[timestamp_index].text)
        if timestamp_str:
            timestamp = datetime.strptime(timestamp_str,"%Y-%m-%d%H:%M:%S")
            if max_old_school_timestamp is None or timestamp > max_old_school_timestamp:
                max_old_school_row = row
                max_old_school_timestamp = timestamp

    links_col = max_old_school_row[links_index]

    cache_link = links_col.find('a', { 'href': re.compile(r'\/caches\/runescape\/(\d+)\/disk.zip') }).get('href')
    xtea_link = links_col.find('a', { 'href': re.compile(r'\/caches\/runescape\/(\d+)\/keys.json') }).get('href')

    os.makedirs(CACHE_DIR, exist_ok=True)

    xtea_file_json = requests.get(CACHES_BASE_URL + xtea_link, allow_redirects=True).json()

    output = []

    for region in xtea_file_json:
        output.append({
            "region": region["mapsquare"],
            "keys": region["key"]
        })

    with open(XTEA_FILE, 'w') as f:
        json.dump(output, f,  indent=4)

    cache_zip = os.path.join(CACHE_DIR, "cache.zip")
    cache_content = requests.get(CACHES_BASE_URL + cache_link, allow_redirects=True).content

    with open(cache_zip, 'wb') as f:
        f.write(cache_content)

    with zipfile.ZipFile(cache_zip, "r") as f:
        f.extractall(OUTPUT_DIR)

    os.remove(cache_zip)


def build_full_map_images():
    os.chdir('/runelite/cache')

    subprocess.run(['git', 'pull'], check=True)

    subprocess.run(['mvn', '-Dmaven.test.skip', '-DskipTests', 'package'], check=True)

    jar_file = glob.glob("target/*jar-with-dependencies.jar")[0]

    subprocess.run(
        [
            'java', 
            '-Xmx8g', 
            '-cp', 
            jar_file,
            'net.runelite.cache.MapImageDumper', 
            CACHE_DIR, 
            XTEA_FILE, 
            GENERATED_FULL_IMAGES
        ], 
        check=True
    )


def get_changed_tiles(old_image, new_image, zoom):
    # We assume here that both images are the same size.
    img_width_px = new_image.width
    img_height_px = new_image.height

    changed_tiles = []

    for tile_x in range(0, img_width_px, TILE_SIZE_PX):
        for tile_y in range(0, img_height_px, TILE_SIZE_PX):
            old_image_tile = old_image.crop(tile_x, tile_y, TILE_SIZE_PX, TILE_SIZE_PX)
            new_image_tile = new_image.crop(tile_x, tile_y, TILE_SIZE_PX, TILE_SIZE_PX)

            old_image_buff = old_image_tile.write_to_memory()
            new_image_buff = new_image_tile.write_to_memory()

            old_image_np = np.ndarray(buffer=old_image_buff,
                                      dtype=np.uint8,
                                      shape=[old_image_tile.height, old_image_tile.width, old_image_tile.bands])

            new_image_np = np.ndarray(buffer=new_image_buff,
                                      dtype=np.uint8,
                                      shape=[new_image_tile.height, new_image_tile.width, new_image_tile.bands])

            ssim = compare_ssim(old_image_np, new_image_np, multichannel=True)

            has_changed = ssim < 0.999

            if has_changed:
                x = int(tile_x / TILE_SIZE_PX)
                y = int(tile_y / TILE_SIZE_PX)
                max_y = math.floor(new_image.height / TILE_SIZE_PX)
                y = max_y - y - 1

                changed_tiles.append({
                    "pixel_x": tile_x,
                    "pixel_y": tile_y,
                    "x": x,
                    "y": y,
                    "image": new_image_tile
                })

    return changed_tiles


def split_tiles_to_new_zoom(changed_tiles, plane, new_zoom):
    new_changed_tiles = []

    for changed_tile in changed_tiles:
        original_x = changed_tile["x"]
        original_y = changed_tile["y"]

        tile_image = changed_tile["image"]

        tile_image_resized = tile_image.resize(2, kernel='nearest')

        new_x = original_x * 2
        new_y = original_y * 2

        tile_image_0 = tile_image_resized.crop(0, TILE_SIZE_PX, TILE_SIZE_PX, TILE_SIZE_PX)
        new_changed_tiles.append({
            "x": new_x,
            "y": new_y,
            "image": tile_image_0
        })
        save_tile(tile_image_0, plane, new_zoom, new_x, new_y)

        tile_image_1 = tile_image_resized.crop(TILE_SIZE_PX, TILE_SIZE_PX, TILE_SIZE_PX, TILE_SIZE_PX)
        new_changed_tiles.append({
            "x": new_x + 1,
            "y": new_y,
            "image": tile_image_1
        })
        save_tile(tile_image_1, plane, new_zoom, new_x + 1, new_y)

        tile_image_2 = tile_image_resized.crop(0, 0, TILE_SIZE_PX, TILE_SIZE_PX)
        new_changed_tiles.append({
            "x": new_x,
            "y": new_y + 1,
            "image": tile_image_2
        })
        save_tile(tile_image_2, plane, new_zoom, new_x, new_y + 1)

        tile_image_3 = tile_image_resized.crop(TILE_SIZE_PX, 0, TILE_SIZE_PX, TILE_SIZE_PX)
        new_changed_tiles.append({
            "x": new_x + 1,
            "y": new_y + 1,
            "image": tile_image_3
        })
        save_tile(tile_image_3, plane, new_zoom, new_x + 1, new_y + 1)

    return new_changed_tiles


def join_tiles_to_new_zoom(changed_tiles, plane, current_zoom, new_zoom):
    new_changed_tiles = []

    for changed_tile in changed_tiles:
        original_x = changed_tile["x"]
        original_y = changed_tile["y"]

        is_left = original_x % 2 == 0
        is_bottom = original_y % 2 == 0

        if is_left:
            side = Side.BOTTOM_LEFT if is_bottom else Side.TOP_LEFT
        else:
            side = Side.BOTTOM_RIGHT if is_bottom else Side.TOP_RIGHT

        if side == Side.TOP_LEFT:
            tiles = [
                load_tile(plane, current_zoom, original_x, original_y),
                load_tile(plane, current_zoom, original_x + 1, original_y),
                load_tile(plane, current_zoom, original_x, original_y - 1),
                load_tile(plane, current_zoom, original_x + 1, original_y - 1)
            ]
        elif side == Side.TOP_RIGHT:
            tiles = [
                load_tile(plane, current_zoom, original_x - 1, original_y),
                load_tile(plane, current_zoom, original_x, original_y),
                load_tile(plane, current_zoom, original_x - 1, original_y - 1),
                load_tile(plane, current_zoom, original_x, original_y - 1)
            ]
        elif side == Side.BOTTOM_LEFT:
            tiles = [
                load_tile(plane, current_zoom, original_x, original_y + 1),
                load_tile(plane, current_zoom, original_x + 1, original_y + 1),
                load_tile(plane, current_zoom, original_x, original_y),
                load_tile(plane, current_zoom, original_x + 1, original_y)
            ]
        else:
            tiles = [
                load_tile(plane, current_zoom, original_x - 1, original_y + 1),
                load_tile(plane, current_zoom, original_x, original_y + 1),
                load_tile(plane, current_zoom, original_x - 1, original_y),
                load_tile(plane, current_zoom, original_x, original_y)
            ]

        new_tile_image = pyvips.Image.arrayjoin(tiles, across=2)

        new_tile_image_resized = new_tile_image.resize(0.5, kernel='lanczos3')

        new_x = math.floor(original_x / 2)
        new_y = math.floor(original_y / 2)

        save_tile(new_tile_image_resized, plane, new_zoom, new_x, new_y)

        new_changed_tiles.append({
            "x": new_x,
            "y": new_y,
            "image": new_tile_image_resized
        })

    return new_changed_tiles


def save_tile(tile_image, plane, zoom, x, y):
    file_dir = Path(TILE_DIR, str(plane), str(zoom), str(x))

    file_dir.mkdir(parents=True, exist_ok=True)

    file_path = Path(file_dir, str(y) + ".png")

    tile_image.pngsave(str(file_path), compression=9)


def load_tile(plane, zoom, x, y):
    file_path = Path(TILE_DIR, str(plane), str(zoom), str(x), str(y) + ".png")

    if not os.path.isfile(file_path):
        existing_tile_path = Path(REPO_DIR, str(plane), str(zoom), str(x), str(y) + ".png")

        if not os.path.isfile(existing_tile_path):
            image = pyvips.Image.black(TILE_SIZE_PX, TILE_SIZE_PX, bands=4)
            image = image.draw_rect([0, 0, 0, 0], 0, 0, TILE_SIZE_PX, TILE_SIZE_PX, fill=True)
            return image
        else:
            path = existing_tile_path
    else:
        path = file_path

    image = pyvips.Image.new_from_file(str(path), access="sequential")

    if not image.hasalpha():
        image = image.addalpha()

    return image


def output_tile_diff_image(changed_tiles, new_image_location, output_file_name):

    diff_img = im = Image.open(new_image_location)

    draw = ImageDraw.Draw(im)

    for tile in changed_tiles:
        shape = [(tile["pixel_x"], tile["pixel_y"]), (tile["pixel_x"] + TILE_SIZE_PX, tile["pixel_y"] + TILE_SIZE_PX)]
        draw.rectangle(shape, fill=None, outline="red", width=3)

    diff_img.save(output_file_name)


if __name__ == '__main__':
    main()
