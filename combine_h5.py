#!/usr/bin/env python3
import argparse
import glob
import logging
import os

import h5py
import errno
from tqdm import tqdm


def _copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _ensure_group(out_parent, name, src_group):
    if name in out_parent:
        out_group = out_parent[name]
    else:
        out_group = out_parent.create_group(name)
        _copy_attrs(src_group, out_group)
    return out_group


def _create_output_dataset(out_group, name, src_dset):
    src_shape = src_dset.shape
    if src_shape == ():
        out_dset = out_group.create_dataset(name, data=src_dset[()])
        _copy_attrs(src_dset, out_dset)
        return out_dset

    maxshape = (None,) + src_shape[1:]
    chunks = src_dset.chunks
    if chunks is None:
        chunks = (min(1024, src_shape[0]),) + src_shape[1:]

    out_dset = out_group.create_dataset(
        name,
        shape=(0,) + src_shape[1:],
        maxshape=maxshape,
        dtype=src_dset.dtype,
        chunks=chunks,
        compression=src_dset.compression,
        compression_opts=src_dset.compression_opts,
        shuffle=src_dset.shuffle,
        fletcher32=src_dset.fletcher32,
        fillvalue=src_dset.fillvalue,
    )
    _copy_attrs(src_dset, out_dset)
    return out_dset


def _append_dataset(out_dset, src_dset, chunk_size):
    src_shape = src_dset.shape
    if src_shape == ():
        src_value = src_dset[()]
        if out_dset[()] != src_value:
            logging.warning("Scalar dataset differs: %s", src_dset.name)
        return

    if out_dset.shape[1:] != src_shape[1:]:
        raise ValueError(
            "Shape mismatch for %s: output %s vs input %s"
            % (src_dset.name, out_dset.shape, src_shape)
        )

    old_len = out_dset.shape[0]
    new_len = old_len + src_shape[0]
    out_dset.resize((new_len,) + out_dset.shape[1:])

    for start in range(0, src_shape[0], chunk_size):
        end = min(start + chunk_size, src_shape[0])
        out_dset[old_len + start : old_len + end] = src_dset[start:end]


def _merge_group(src_group, out_group, chunk_size):
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            next_out = _ensure_group(out_group, name, obj)
            _merge_group(obj, next_out, chunk_size)
        elif isinstance(obj, h5py.Dataset):
            if name in out_group:
                out_dset = out_group[name]
            else:
                out_dset = _create_output_dataset(out_group, name, obj)
            _append_dataset(out_dset, obj, chunk_size)


def combine_h5(input_dir, output_path, pattern, chunk_size):
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        logging.warning("No input files found in %s with pattern %s — skipping.", input_dir, pattern)
        return False

    with h5py.File(output_path, "w") as out_file:
        for path in tqdm(paths, desc="Processing files"):
            logging.info("Processing %s", path)
            with h5py.File(path, "r") as in_file:
                _merge_group(in_file, out_file, chunk_size)
    return True


def _remove_sig0_files(input_dir, pattern):
    """Remove files matching sig_*0.h5 in `input_dir` (no recursion).
    """
    pattern = os.path.join(input_dir, pattern)
    paths = sorted(glob.glob(pattern))
    
    removed = []
    for path in paths:
        try:
            os.remove(path)
            removed.append(path)
            logging.info("Removed %s", path)
        except OSError as e:
            if e.errno == errno.ENOENT:
                continue
            logging.warning("Failed to remove %s: %s", path, e)
    return removed


def main():
    parser = argparse.ArgumentParser(
        description="Combine HDF5 files by concatenating datasets along axis 0."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing .h5/.hdf5 files or a directory that contains train/test/val subdirectories.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        help=(
            "Output HDF5 file (optional). If omitted and `input_dir` contains "
            "train/test/val subdirectories, the script will combine files in each "
            "subdirectory and write {subdir}/sig_combined_{subdir}.h5."
        ),
    )
    parser.add_argument(
        "--pattern",
        default="sig_[0-9]*.h5",
        help="Glob pattern for input files (default: sig_[0-9]*.h5).",
    )
    parser.add_argument(
        "--out_prefix",
        default="sig_combined",
        help="Prefix for output combined files (default: sig_combined).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Rows to read/write per chunk (default: 1024).",
    )
    parser.add_argument(
        "--no-delete",
        dest="no_delete",
        action="store_true",
        default=False,
        help="Keep intermediate files after combining (default: delete them).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable info logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    # If output is provided, behave as before (combine files in a single directory).
    if args.output:
        combined = combine_h5(args.input_dir, args.output, args.pattern, args.chunk_size)
        if combined and not args.no_delete:
            _remove_sig0_files(args.input_dir, args.pattern)
        return

    # Otherwise, expect `input_dir` to contain train/test/val subdirectories.
    # Subdirectories that are missing or have no matching files are skipped.
    subsets = ["train", "test", "val"]
    combined_splits, skipped_splits = [], []

    for s in subsets:
        in_dir = os.path.join(args.input_dir, s)
        if not os.path.isdir(in_dir):
            print(f"[combine_h5] SKIP  {s}: directory not found ({in_dir})")
            skipped_splits.append(s)
            continue
        out_path = os.path.join(in_dir, f"{args.out_prefix}_{s}.h5")
        print(f"[combine_h5] START {s}: {in_dir} -> {out_path}")
        combined = combine_h5(in_dir, out_path, args.pattern, args.chunk_size)
        if combined:
            print(f"[combine_h5] DONE  {s}: wrote {out_path}")
            combined_splits.append(s)
            if not args.no_delete:
                _remove_sig0_files(in_dir, args.pattern)
        else:
            print(f"[combine_h5] SKIP  {s}: no matching files in {in_dir}")
            skipped_splits.append(s)

    print(f"[combine_h5] combined={combined_splits}  skipped={skipped_splits}")


if __name__ == "__main__":
    main()