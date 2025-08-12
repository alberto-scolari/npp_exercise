
# NPP Exercise
This is an exercise to show NPP capabilities.

# Quickstart
For the basic functionalities, a [`Makefile`](Makefile) is available.
To compile and run on a single image, you may run

```bash
make test
```

To compile and run on an entire set of images, you may run

```bash
make test_dir
```

This command:

1. builds the executable `edge_detect.x`
2. creates the directory `images/input`
3. extracts the PGM images inside `images/images.tar.gz` into `images/input`
4. creates the directory `images/output`
5. runs `edge_detect.x` in the directory `images/input` with output into `images/output`

To just compile

```bash
make build
```

To clean everything

```bash
make distclean
```

To clean only the built binaries

```bash
make clean
```

To see the synopsis

```bash
$ ./edge_detect.x --help
Usage: edge_detect [--help] [-o VAR] [--batch VAR] [--dir] input

Positional arguments:
  input       input file or directory (see '--dir' option)

Optional arguments:
  -h, --help  shows help message and exits
  -o          output file or directory (depending on input) [nargs=0..1] [default: "."]
  --batch     batch size (none means decide from hardware)
  --dir       input is a directory
```

## Code structure and organization

The structure of the repository follows the [C++ Canonical Project Structure](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1204r0.html).

The code is formatted via `make clang-format` and checked via `make clang-tidy`.

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
