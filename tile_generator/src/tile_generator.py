"""
Tile generator for creating tile sets for OSRS maps to be used with map viewers such as leaflet.
"""

import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from enum import Enum
from multiprocessing import Pool
from pathlib import Path

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed, thread
import concurrent.futures
import numpy as np
import pyvips
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

CACHES_BASE_URL = "https://archive.openrs2.org"

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

Image.MAX_IMAGE_PIXELS = 1000000000000000

TILE_SIZE_PX = 256

MIN_ZOOM = 3
MAX_ZOOM = 11

MIN_Z = 0
MAX_Z = 3

REPO_DIR = '/repo' # Name of the directory mounted on the local machine
ROOT_CACHE_DIR = os.path.join(REPO_DIR, 'cache/')
GENERATED_FULL_IMAGES = os.path.join(REPO_DIR, 'generated_images/')
TILE_DIR = REPO_DIR

image_prefix = "full_image_"

logging.basicConfig(
    format='%(asctime)s %(levelname)-4s %(message)s',
    level=logging.WARNING,
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_TILE_IMAGE = (
    pyvips.Image.black(TILE_SIZE_PX, TILE_SIZE_PX, bands=4)
    .draw_rect([0, 0, 0, 0], 0, 0, TILE_SIZE_PX, TILE_SIZE_PX, fill=True)
)

class Side(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4


def main():
    LOG.info("Downloading cache & XTEAs")
    [cache_dir, xtea_file] = download_cache_with_xteas()

    LOG.info("Building map base images")
    build_full_map_images(cache_dir, xtea_file)

    LOG.info("Generating tiles")
    for plane in range(MAX_Z + 1):
        generate_tiles_for_plane(plane)

    for plane in range(MIN_Z, MAX_Z + 1):
        previous_map_image_name = os.path.join(GENERATED_FULL_IMAGES, f"previous-map-image-{plane}.png")
        current_map_image_name = os.path.join(GENERATED_FULL_IMAGES, f"current-map-image-{plane}.png")
        generated_file_name = os.path.join(GENERATED_FULL_IMAGES, f"new-map-image-{plane}.png")

        os.replace(current_map_image_name, previous_map_image_name)
        os.replace(generated_file_name, current_map_image_name)
       

def download_cache_with_xteas():
    """
        Downloads latest cache and XTEAs from specified URL.
        These are both hard requirements for generating the full OSRS map image with Runelite
    """

    latest_cache_version = fetch_latest_osrs_cache_version()

    LOG.info(f"Cache version: {latest_cache_version['links']['base']}")
    LOG.info(f"Cache upload date: {latest_cache_version['timestamp']}")
    LOG.info(f"Cache build: {latest_cache_version['build(s)']}")
    
    cache_dir = os.path.join(ROOT_CACHE_DIR, latest_cache_version['timestamp'].strftime("%Y-%m-%d_%H_%M_%S") + "/")
    xtea_file = os.path.join(cache_dir, "xteas.json")

    if os.path.isdir(cache_dir):
        print(f"Cache directory already exists, skipping download")
    else:
        os.makedirs(cache_dir, exist_ok=True)
        download_xteas(latest_cache_version["links"]["xteas"], xtea_file)
        download_and_extract_cache(latest_cache_version["links"]["cache"], cache_dir)

    return [cache_dir, xtea_file]


def fetch_latest_osrs_cache_version():
    """
        Returns the latest OSRS cache version based on upload timestamp
    """
    cache_versions = fetch_osrs_cache_versions()
    latest_cache_version = max(cache_versions, key=lambda cache_version: cache_version["timestamp"])
    return latest_cache_version
    

def fetch_osrs_cache_versions():
    """
        Returns a list of OSRS cache version with upload timestamp
        by scaping https://archive.openrs2.org
    """
    caches = requests.get(CACHES_BASE_URL + "/caches", allow_redirects=True)
    soup = BeautifulSoup(caches.content, features="html.parser")
    cache_table = soup.find('table')
    table_body = cache_table.find('tbody')
    rows = table_body.find_all('tr')

    header = cache_table.find("thead")
    header_cols = header.find_all('th')
    header_texts = [ col.text.strip().lower() for col in header_cols ]

    oldschool_rows = []
    for row in rows:
        columns = row.find_all('td')
        mapped_row = {}

        for i in range(len(columns)):
            column = columns[i]
            header_text = header_texts[i]
            mapped_row[header_text] = column

        if mapped_row["game"].text.strip().lower() == "oldschool":
            timestamp_str = re.sub(r'\s+', '', mapped_row["timestamp"].text)

            if not timestamp_str:
                continue

            timestamp = datetime.strptime(timestamp_str,"%Y-%m-%d%H:%M:%S")
            mapped_row["timestamp"] = timestamp

            links_col = mapped_row["links"]

            cache_link = links_col.find('a', { 'href': re.compile(r'\/caches\/runescape\/(\d+)\/disk.zip') }).get('href')

            if not cache_link:
                raise ValueError("Failed to extract cache version from HTML")

            cache_idx_num = re.match(r'\/caches\/runescape\/(\d+)\/disk.zip', cache_link).group(1) 

            mapped_row["links"] = {
                "base": f"{CACHES_BASE_URL}/caches/runescape/{cache_idx_num}",
                "cache": f"{CACHES_BASE_URL}/caches/runescape/{cache_idx_num}/disk.zip",
                "xteas": f"{CACHES_BASE_URL}/caches/runescape/{cache_idx_num}/keys.json"
            }

            if mapped_row["build(s)"]:
                mapped_row["build(s)"] = [build.replace(" ", "") for build in mapped_row["build(s)"].text.strip().split("\n")]

            oldschool_rows.append(mapped_row)

    return oldschool_rows


def download_xteas(xtea_url, output_file):
    """
        Downloads XTEAs in the format required by Runelite
    """
    xtea_file_json = requests.get(xtea_url, allow_redirects=True).json()

    runelite_xteas = map_openrs2_xteas_to_runelite_format(xtea_file_json)

    with open(output_file, 'w') as f:
        json.dump(runelite_xteas, f,  indent=4)
    

def map_openrs2_xteas_to_runelite_format(xteas): 
    return [
        {
            "region": region["mapsquare"],
            "keys": region["key"]
        }
        for region in xteas
    ]


def download_and_extract_cache(cache_zip_url, output_dir):
    cache_zip = os.path.join(output_dir, "cache.zip")
    cache_content = requests.get(cache_zip_url, allow_redirects=True).content

    with open(cache_zip, 'wb') as f:
        f.write(cache_content)

    with zipfile.ZipFile(cache_zip, "r") as zip_file:
        for name in zip_file.namelist():
            basename = os.path.basename(name)
            
            if not basename:
                continue

            member = zip_file.open(name)
            with open(os.path.join(output_dir, basename), 'wb') as outfile:
                shutil.copyfileobj(member, outfile)

    os.remove(cache_zip)


def build_full_map_images(cache_dir, xtea_file):
    """
        Runs Runelite's MapImageDumper Java program to generate full OSRS map images
    """
    os.chdir('/runelite/cache')

    jar_file = glob.glob("target/*jar-with-dependencies.jar")[0]

    subprocess.run(
        [
            'java', 
            '-Xmx8g', 
            '-cp', 
            jar_file,
            'net.runelite.cache.MapImageDumper', 
            '--cachedir', cache_dir, 
            '--xteapath', xtea_file, 
            '--outputdir', GENERATED_FULL_IMAGES
        ], 
        check=True
    )

    for plane in range(MIN_Z, MAX_Z + 1):
        new_map_image_path = os.path.join(GENERATED_FULL_IMAGES, f"img-{plane}.png")
        renamed_new_map_image_path = os.path.join(GENERATED_FULL_IMAGES, f"new-map-image-{plane}.png")
        os.replace(new_map_image_path, renamed_new_map_image_path)


def generate_tiles_for_plane(plane):
    """
        Generates OSRS map tiles for a given Z plane.
        
        The tiles are generated from the full OSRS map image by:
            1. Finding which "zoom" level the full image is in
            2. Splitting the full image into tiles at each higher zoom level from the original (e.g. 8 -> 9 -> 10 -> 11)
            3. Joining split tiles in the original zoom level to lower zoom levels (e.g. 8 -> 7 -> 6)

        This process is optimised by first diffing the new full OSRS map image with the previously generated OSRS Map image for the previous cache version.
        We use pyvips to compare each tile at the original zoom level (e.g. 8) with the tile at the same coordinates at the same zoom level in the old image.
        We then only split & join tiles which have changed, to higher or lower zoom levels. This massively reduces the output size & run time of the program.

        In an ideal world we would load the new cache & old cache (with XTEAs) and compare the underlying data for a given region to see if tiles have changed.
        However I'm too lazy to implement that right now, so instead we're just comparing images.

        Where possible we also utilise threadpools to speed up the process.
    """
    log_prefix = f"[Plane: {plane}]:"

    LOG.info(f"{log_prefix} Generating plane {plane}")
    LOG.info(f"{log_prefix} Loading images into memory")

    old_image_location = os.path.join(GENERATED_FULL_IMAGES, f"current-map-image-{plane}.png")
    new_image_location = os.path.join(GENERATED_FULL_IMAGES, f"new-map-image-{plane}.png")

    old_image = pyvips.Image.new_from_file(old_image_location)
    new_image = pyvips.Image.new_from_file(new_image_location)

    image_width = new_image.width
    image_width_tiles = int(image_width / TILE_SIZE_PX)

    starting_zoom = int(math.sqrt(math.pow(2, math.ceil(math.log(image_width_tiles) / math.log(2)))))

    LOG.info(f"{log_prefix} Calculating changed tiles")
    changed_tiles = get_changed_tiles(old_image, new_image, plane, starting_zoom)

    LOG.info(f"{log_prefix} Storing diff image")
    output_tile_diff_image(changed_tiles, new_image_location,  str(Path(GENERATED_FULL_IMAGES, f"diff-map-image-{plane}.png")))

    LOG.info(f"{log_prefix} Found {len(changed_tiles)} changed tiles at zoom level {starting_zoom}")

    LOG.info(f"{log_prefix} Saving changed tiles at zoom level {starting_zoom}")
    for tile in changed_tiles:
        save_tile(tile["image"], plane, starting_zoom, tile["x"], tile["y"])

    next_changed_tiles = changed_tiles
    for zoom in range(starting_zoom + 1, MAX_ZOOM + 1):
        LOG.info(f"{log_prefix} Splitting changed tiles from zoom level {zoom-1} to zoom level {zoom}")
        next_changed_tiles = split_tiles_to_new_zoom(next_changed_tiles, plane, zoom)
        LOG.info(f"{log_prefix} Done")

    for zoom in reversed(range(MIN_ZOOM + 1, starting_zoom + 1)):
        LOG.info(f"{log_prefix} Joining changed tiles from zoom level {zoom} to zoom level {zoom - 1}")
        changed_tiles = join_tiles_to_new_zoom(changed_tiles, plane, zoom, zoom - 1)
        LOG.info(f"{log_prefix} Done")


def get_changed_tiles(old_image, new_image, plane, zoom):
    new_image_width_px = new_image.width
    new_image_height_px = new_image.height

    changed_tiles = []

    with thread_pool_executor() as executor:
        futures = []

        # Loop over all tiles in the new image
        for tile_x in range(0, new_image_width_px, TILE_SIZE_PX):
            for tile_y in range(0, new_image_height_px, TILE_SIZE_PX):
                futures.append(
                    executor.submit(
                        has_tile_changed,
                        plane=plane,
                        zoom=zoom,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        old_image=old_image,
                        new_image=new_image
                    )
                )

        for future in concurrent.futures.as_completed(futures):
            ((tile_x, tile_y), new_image_tile, has_changed) = future.result()

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


def has_tile_changed(plane, zoom, tile_x, tile_y, old_image, new_image):
    new_image_tile = new_image.crop(tile_x, tile_y, TILE_SIZE_PX, TILE_SIZE_PX)

    # If there is no tile at (tile_x, tile_y) in the old image
    if tile_x > old_image.width - TILE_SIZE_PX or tile_y > old_image.height - TILE_SIZE_PX:
        return ((tile_x, tile_y), new_image_tile, True)

    old_image_tile = old_image.crop(tile_x, tile_y, TILE_SIZE_PX, TILE_SIZE_PX)

    old_image_buff = old_image_tile.write_to_memory()
    new_image_buff = new_image_tile.write_to_memory()

    old_image_np = np.ndarray(buffer=old_image_buff,
                            dtype=np.uint8,
                            shape=[old_image_tile.height, old_image_tile.width, old_image_tile.bands])

    new_image_np = np.ndarray(buffer=new_image_buff,
                            dtype=np.uint8,
                            shape=[new_image_tile.height, new_image_tile.width, new_image_tile.bands])

    ssim = structural_similarity(old_image_np, new_image_np, multichannel=True)

    has_changed = ssim < 0.999

    return ((tile_x, tile_y), new_image_tile, has_changed)


def split_tiles_to_new_zoom(changed_tiles, plane, new_zoom):
    new_changed_tiles = []

    with thread_pool_executor() as executor:
        futures = []

        for changed_tile in changed_tiles:
            futures.append(
                executor.submit(
                    split_tile_to_new_zoom,
                    changed_tile=changed_tile,
                    plane=plane,
                    new_zoom=new_zoom
                )
            )

        for future in concurrent.futures.as_completed(futures):
            new_changed_tiles.extend(future.result())

    return new_changed_tiles


def split_tile_to_new_zoom(changed_tile, plane, new_zoom):
    original_x = changed_tile["x"]
    original_y = changed_tile["y"]

    tile_image = changed_tile["image"]

    tile_image_resized = tile_image.resize(2, kernel='nearest')

    new_x = original_x * 2
    new_y = original_y * 2

    tile_image_0 = tile_image_resized.crop(0, TILE_SIZE_PX, TILE_SIZE_PX, TILE_SIZE_PX)
    save_tile(tile_image_0, plane, new_zoom, new_x, new_y)

    tile_image_1 = tile_image_resized.crop(TILE_SIZE_PX, TILE_SIZE_PX, TILE_SIZE_PX, TILE_SIZE_PX)
    save_tile(tile_image_1, plane, new_zoom, new_x + 1, new_y)

    tile_image_2 = tile_image_resized.crop(0, 0, TILE_SIZE_PX, TILE_SIZE_PX)
    save_tile(tile_image_2, plane, new_zoom, new_x, new_y + 1)

    tile_image_3 = tile_image_resized.crop(TILE_SIZE_PX, 0, TILE_SIZE_PX, TILE_SIZE_PX)
    save_tile(tile_image_3, plane, new_zoom, new_x + 1, new_y + 1)

    # New tiles at new zoom
    return [
        {
            "x": new_x,
            "y": new_y,
            "image": tile_image_0
        },
        {
            "x": new_x + 1,
            "y": new_y,
            "image": tile_image_1
        },
        {
            "x": new_x,
            "y": new_y + 1,
            "image": tile_image_2
        },
        {
            "x": new_x + 1,
            "y": new_y + 1,
            "image": tile_image_3
        }
    ]


def join_tiles_to_new_zoom(changed_tiles, plane, current_zoom, new_zoom):
    new_changed_tiles = []

    with thread_pool_executor() as executor:
        futures = []

        for changed_tile in changed_tiles:
            futures.append(
                executor.submit(
                    join_changed_tile_to_new_zoom,
                    changed_tile=changed_tile,
                    plane=plane,
                    current_zoom=current_zoom,
                    new_zoom=new_zoom
                )
            )

        for future in concurrent.futures.as_completed(futures):
            new_changed_tiles.append(future.result())

    return new_changed_tiles


def join_changed_tile_to_new_zoom(changed_tile, plane, current_zoom, new_zoom):
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
            load_generated_tile(plane, current_zoom, original_x, original_y),
            load_generated_tile(plane, current_zoom, original_x + 1, original_y),
            load_generated_tile(plane, current_zoom, original_x, original_y - 1),
            load_generated_tile(plane, current_zoom, original_x + 1, original_y - 1)
        ]
    elif side == Side.TOP_RIGHT:
        tiles = [
            load_generated_tile(plane, current_zoom, original_x - 1, original_y),
            load_generated_tile(plane, current_zoom, original_x, original_y),
            load_generated_tile(plane, current_zoom, original_x - 1, original_y - 1),
            load_generated_tile(plane, current_zoom, original_x, original_y - 1)
        ]
    elif side == Side.BOTTOM_LEFT:
        tiles = [
            load_generated_tile(plane, current_zoom, original_x, original_y + 1),
            load_generated_tile(plane, current_zoom, original_x + 1, original_y + 1),
            load_generated_tile(plane, current_zoom, original_x, original_y),
            load_generated_tile(plane, current_zoom, original_x + 1, original_y)
        ]
    else:
        tiles = [
            load_generated_tile(plane, current_zoom, original_x - 1, original_y + 1),
            load_generated_tile(plane, current_zoom, original_x, original_y + 1),
            load_generated_tile(plane, current_zoom, original_x - 1, original_y),
            load_generated_tile(plane, current_zoom, original_x, original_y)
        ]

    new_tile_image = pyvips.Image.arrayjoin(tiles, across=2)

    new_tile_image_resized = new_tile_image.resize(0.5, kernel='lanczos3')

    new_x = math.floor(original_x / 2)
    new_y = math.floor(original_y / 2)

    save_tile(new_tile_image_resized, plane, new_zoom, new_x, new_y)

    return {
        "x": new_x,
        "y": new_y,
        "image": new_tile_image_resized
    }

def save_tile(tile_image, plane, zoom, x, y):
    file_dir = Path(TILE_DIR, str(plane), str(zoom), str(x))

    file_dir.mkdir(parents=True, exist_ok=True)

    file_path = Path(file_dir, str(y) + ".png")

    tile_image.pngsave(str(file_path), compression=9)


def load_generated_tile(plane, zoom, x, y):
    """
        Loads a tile image from the tile set stored in the repo
    """
    if not generated_tile_exists(plane, zoom, x, y):
        return DEFAULT_TILE_IMAGE

    file_path = generated_tile_path(plane, zoom, x, y)

    image = pyvips.Image.new_from_file(str(file_path), access="sequential")

    if not image.hasalpha():
        image = image.addalpha()

    return image


def generated_tile_exists(plane, zoom, x, y):
    return os.path.isfile(generated_tile_path(plane, zoom, x, y))


def generated_tile_path(plane, zoom, x, y):
    return Path(TILE_DIR, str(plane), str(zoom), str(x), str(y) + ".png")


def output_tile_diff_image(changed_tiles, new_image_location, output_file_name):
    diff_image = im = Image.open(new_image_location)

    draw = ImageDraw.Draw(im)

    for tile in changed_tiles:
        shape = [(tile["pixel_x"], tile["pixel_y"]), (tile["pixel_x"] + TILE_SIZE_PX, tile["pixel_y"] + TILE_SIZE_PX)]
        draw.rectangle(shape, fill=None, outline="red", width=3)

    diff_image.save(output_file_name)


@contextmanager
def thread_pool_executor():
    with ThreadPoolExecutor() as executor:
        try:
            yield executor
        except KeyboardInterrupt:
            LOG.warning("Performing non-graceful shutdown of threads")
            executor._threads.clear()
            concurrent.futures.thread._threads_queues.clear()


if __name__ == '__main__':
    main()
