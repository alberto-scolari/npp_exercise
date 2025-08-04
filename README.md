
# NPP Exercise
This is an exercise to show NPP capabilities.

## Inspect the output
To inspect the output, you may want to convert it from PGM to PNG using [ImageMagick](https://imagemagick.org)

```bash
magick <file name>.pgm <file name>.png
```

## Dependencies and Licenses
The images inside `images/images.tar.gz` (used as input) were downloaded from https://sipi.usc.edu/database/database.php?volume=sequences (where they are freely available) and converted to PGM in order to avoid flooding warnings.

> [!NOTE]
> To convert each image, the following command was used
> ```bash
> magick <file name>.tiff -depth 8 <file name>.pgm
> ```

The [`argparse`](https://github.com/p-ranav/argparse) library is available under [MIT License](https://github.com/p-ranav/argparse/blob/master/LICENSE).
