# osrs_map_tiles

The OSRS map split into tiles for use with map viewers.

## Generating tiles

1. Install docker: https://docs.docker.com/get-docker/
2. Update your docker settings to set the maximum memory to 8GB
2. Open powershell in windows, or the terminal in other OS'
3. From the root directory of this repo run

### Windows
```
$Env:DOCKER_BUILDKIT=0
docker build ./tile_generator -t "map-tile-generator"
docker run -it -v "${pwd}:/repo" map-tile-generator
```

### Mac / Unix
```
export DOCKER_BUILDKIT=0
docker build ./tile_generator -t "map-tile-generator"
docker run -it -v "${pwd}:/repo" map-tile-generator
```