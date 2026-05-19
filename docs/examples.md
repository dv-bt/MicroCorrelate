# Examples

## Stitching Maps tilesets

The {func}`~microcorrelate.stitching.stitch_images` function assembles tiled datasets
exported by the Thermo Fisher Maps software into a single image file.
Maps pre-stitches the tiles so there is no overlap to resolve; the function
simply places each tile at the correct position, crops any empty borders, and
writes physical metadata (pixel size and stage position) into the output file.

### Save as TIFF

The simplest call writes a single BigTIFF file with OME metadata:

```python
from pathlib import Path
from microcorrelate.stitching import stitch_images

tileset_path = Path("data/LayersData/tileset")
dest_path = Path("out/stitched.tif")
dest_path.parent.mkdir(exist_ok=True)

stitch_images(tileset_path, dest_path)
```

The output TIFF embeds physical pixel size and, when a `MapsProject.xml` is
found, the absolute stage position as OME `Plane` coordinates.

:::{tip}
Pass `verbose=True` to print progress messages during stitching, which is
useful for large datasets.
:::

### Save as OME-Zarr

For large datasets or workflows that benefit from multi-resolution access,
save as an [OME-NGFF](https://ngff.openmicroscopy.org/) Zarr store:

```python
stitch_images(
    tileset_path,
    Path("out/stitched.zarr"),
    pyramid_levels=4,   # build 4 resolution levels
    verbose=True,
)
```

Scale and translation coordinate transformations are written in nanometres,
following the OME-NGFF v0.5 spec. If the stage position cannot be determined
(e.g. `MapsProject.xml` is missing), the translation defaults to zero and a
warning is raised.

### Group multiple acquisitions in one store

When correlating acquisitions from several sensors, all can be stored in a single Zarr
archive using the `group_path` argument:

```python
acquisitions = {
    "cbs": Path("path/to/cbs/tileset"),
    "edt": Path("path/to/edt/tileset"),
}

for name, tileset in acquisitions.items():
    stitch_images(
        tileset,
        Path("out/experiment.zarr"),
        group_path=name,
        pyramid_levels=4,
    )
```

### Command-line interface

A convenience script is also available after installation:

```bash
stitch-maps \
    --source data/LayersData/tileset \
    --dest out/stitched.zarr \
    --pyramid_levels 4 \
    --verbose
```

Run `stitch-maps --help` for the full list of options.

### Reading the result

TIFF files can be opened with any OME-TIFF-aware library, such as
[tifffile](https://github.com/cgohlke/tifffile), and metadata can be read using [imageio](https://github.com/imageio/imageio):

```python
from pathlib import Path
import tifffile
from imageio.v3 import immeta

image_path = Path("out/stitched.tif")
image = tifffile.imread(image_path)
metadata = immeta(image_path)
```

Zarr stores follow the [OME-NGFF v0.5](https://ngff.openmicroscopy.org/) spec and can be
read back, for example, with [ngff-zarr](https://ngff-zarr.readthedocs.io/)

```python
import ngff_zarr as nz

multiscales = nz.from_ngff_zarr("out/stitched.zarr")

# Full resolution image (NgffImage dataclass with .data, .scale, .translation)
image = multiscales.images[0]
print(image.scale)        # {'y': 150.0, 'x': 150.0}  (in nm)
print(image.translation)  # {'y': ..., 'x': ...}

# Get a NumPy array
array = image.data.compute()
```

Zarr stores can also be opened interactively in [Napari](https://napari.org/) via
*File → Open*, or programmatically using the
[napari-ome-zarr](https://github.com/ome/napari-ome-zarr) plugin:

```python
import napari

viewer = napari.Viewer()

# Without group_path: point directly to the store
viewer.open("out/stitched.zarr", plugin="napari-ome-zarr")

# With group_path: point to the group, not the store root
viewer.open("out/experiment.zarr/sem", plugin="napari-ome-zarr")
```