import math
import os
from enum import Enum
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import logging
import numpy as np
import pyvips
import glob
import json
import subprocess
from datetime import datetime
import re
import zipfile
import shutil


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
OUTPUT_DIR = os.path.join(REPO_DIR, 'output/')
ROOT_CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache/')
DIFF_DIR = os.path.join(OUTPUT_DIR, 'diff/')
GENERATED_FULL_IMAGES = os.path.join(OUTPUT_DIR, 'generated_images/')
TILE_DIR = REPO_DIR

image_prefix = "full_image_"

logging.basicConfig(
    format='%(asctime)s %(levelname)-4s %(message)s',
    level=logging.WARNING,
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
    [cache_dir, xtea_file] = download_cache()

    LOG.info("Building map base images")
    build_full_map_images(cache_dir, xtea_file)

    LOG.info("Generating tiles")
    for plane in range(MAX_Z + 1):
        generate_tiles_for_plane(plane)
       

def download_cache():
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
    cache_versions = fetch_osrs_cache_versions()
    latest_cache_version = max(cache_versions, key=lambda cache_version: cache_version["timestamp"])
    return latest_cache_version
    

def fetch_osrs_cache_versions():
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


def generate_tiles_for_plane(plane):
    log_prefix = f"[Plane: {plane}]:"

    LOG.info(f"{log_prefix} Generating plane {plane}")
    LOG.info(f"{log_prefix} Loading images into memory")

    old_image_location = os.path.join(REPO_DIR, f"full_image_{plane}.png")
    new_image_location = os.path.join(GENERATED_FULL_IMAGES, f"img-{plane}.png")

    old_image = pyvips.Image.new_from_file(old_image_location)

    new_image = pyvips.Image.new_from_file(new_image_location)

    image_width = new_image.width
    image_width_tiles = int(image_width / TILE_SIZE_PX)

    starting_zoom = int(math.sqrt(math.pow(2, math.ceil(math.log(image_width_tiles) / math.log(2)))))

    LOG.info(f"{log_prefix} Calculating changed tiles")
    changed_tiles = get_changed_tiles(old_image, new_image, starting_zoom)

    LOG.info(f"{log_prefix} Storing diff image")
    output_tile_diff_image(changed_tiles, new_image_location,  str(Path(DIFF_DIR, f"diff_{plane}.png")))

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

    subprocess.run(['cp', new_image_location, old_image_location], check=True)


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

            ssim = structural_similarity(old_image_np, new_image_np, multichannel=True)

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
