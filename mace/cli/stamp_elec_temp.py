#!/usr/bin/env python
"""Stamp the electronic temperature into the info line of each extxyz frame.

The Tel-conditioned datasets encode the electronic temperature in the file name
(``train_Tel4000.xyz``) rather than in the frames, so nothing downstream of the
file system can see it. MACE reads per-graph properties from the extxyz info
line, so the value has to live there before the data loader can pick it up
(``--elec_temp_key``, default ``elec_temp``).

The rewrite is line-based and streaming rather than a parse-and-reserialise
through ASE: every byte that is not the appended key is copied through
unchanged, so REF_energy and REF_forces keep the exact digits they were written
with and 20 MB files cost one pass.

Note that these frames already carry a ``T=`` key. That is the *ionic*
temperature and it varies frame to frame; the electronic temperature is
constant per file and is a different quantity. This script never touches ``T``.

Examples
--------
    # write data/train_Tel4000_stamped.xyz, inferring Tel=4000 from the name
    mace_stamp_elec_temp data/train_Tel4000.xyz

    # rewrite the originals in place
    mace_stamp_elec_temp --in-place data/*_Tel*.xyz

    # a file whose name does not carry the temperature
    mace_stamp_elec_temp --tel 4000 --in-place data/run.xyz
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

# "Tel4000", "Tel_4000", "tel-4000.5"; the separator is optional and the value
# may be a decimal, so files written by a sweep script still resolve.
TEL_IN_NAME = re.compile(r"[Tt]el[_-]?(\d+(?:\.\d+)?)")

DEFAULT_KEY = "elec_temp"


def tel_from_filename(path: Path) -> float:
    """Return the electronic temperature encoded in ``path``'s name."""
    matches = TEL_IN_NAME.findall(path.name)
    if not matches:
        raise ValueError(
            f"cannot infer the electronic temperature from {path.name!r}: "
            f"no 'Tel<number>' in the name. Pass --tel to give it explicitly."
        )
    if len(set(matches)) > 1:
        raise ValueError(
            f"{path.name!r} contains conflicting temperatures {sorted(set(matches))}; "
            f"pass --tel to say which one applies."
        )
    return float(matches[0])


def _key_pattern(key: str) -> re.Pattern:
    # Bounded on the left by start-or-whitespace so a key does not match the
    # tail of a longer one (`elec_temp` inside `ref_elec_temp`).
    return re.compile(rf"(?:(?<=\s)|^){re.escape(key)}=(\S+)")


def stamp_info_line(line: str, key: str, value: float, *, force: bool) -> str:
    """Return ``line`` with ``key=value`` appended, or replaced if present."""
    stripped = line.rstrip("\n")
    existing = _key_pattern(key).search(stripped)
    if existing is not None:
        if not force:
            raise ValueError(
                f"info line already carries {key}={existing.group(1)}; "
                f"pass --force to overwrite it."
            )
        return _key_pattern(key).sub(f"{key}={value!r}", stripped, count=1) + "\n"
    return f"{stripped} {key}={value!r}\n"


def iter_frames(lines: Iterator[str]) -> Iterator[Tuple[str, str, List[str]]]:
    """Yield ``(count_line, info_line, atom_lines)`` for each extxyz frame.

    Frames are located structurally -- a count line, an info line, then exactly
    that many atom lines -- rather than by pattern, so an atom line that happens
    to look like a header cannot be mistaken for one.
    """
    frame_index = 0
    while True:
        try:
            count_line = next(lines)
        except StopIteration:
            return
        if not count_line.strip():  # tolerate blank lines between frames
            continue
        try:
            natoms = int(count_line.strip())
        except ValueError as exc:
            raise ValueError(
                f"frame {frame_index}: expected an atom count, got {count_line.strip()!r}"
            ) from exc
        if natoms < 0:
            raise ValueError(f"frame {frame_index}: negative atom count {natoms}")

        try:
            info_line = next(lines)
        except StopIteration as exc:
            raise ValueError(
                f"frame {frame_index}: file ends before the info line"
            ) from exc

        atom_lines = []
        for atom_index in range(natoms):
            try:
                atom_lines.append(next(lines))
            except StopIteration as exc:
                raise ValueError(
                    f"frame {frame_index}: ends after {atom_index} of {natoms} atoms"
                ) from exc

        yield count_line, info_line, atom_lines
        frame_index += 1


def stamp_file(src: Path, dst: Path, key: str, value: float, *, force: bool) -> int:
    """Stamp ``src`` into ``dst``; returns the number of frames written.

    The write goes to a temporary file in the destination directory and is only
    then renamed over the target, so an interrupted run cannot leave a
    half-rewritten dataset behind -- which matters most for --in-place, where
    the input is also the output.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), suffix=".xyz.tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as out, src.open("r") as handle:
            for count_line, info_line, atom_lines in iter_frames(iter(handle)):
                out.write(count_line)
                try:
                    out.write(stamp_info_line(info_line, key, value, force=force))
                except ValueError as exc:
                    raise ValueError(f"frame {frames}: {exc}") from exc
                out.writelines(atom_lines)
                frames += 1
        tmp.replace(dst)
        tmp = None  # type: ignore[assignment]
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    return frames


def output_path(src: Path, args: argparse.Namespace) -> Path:
    if args.in_place:
        return src
    directory = Path(args.output_dir) if args.output_dir else src.parent
    return directory / f"{src.stem}{args.suffix}{src.suffix}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", type=Path, help="extxyz files to stamp")
    parser.add_argument(
        "--tel",
        type=float,
        default=None,
        help="electronic temperature in K; default is inferred per file from its name",
    )
    parser.add_argument(
        "--key",
        default=DEFAULT_KEY,
        help=f"info key to write (default: {DEFAULT_KEY}, matches --elec_temp_key)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="rewrite each input file instead of writing a stamped copy",
    )
    parser.add_argument(
        "--suffix",
        default="_stamped",
        help="suffix for the copy when not --in-place (default: _stamped)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for stamped copies (default: alongside each input)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the key where a frame already carries it",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    failures = 0
    for src in args.files:
        if not src.is_file():
            logging.error("%s: not a file", src)
            failures += 1
            continue
        try:
            tel = args.tel if args.tel is not None else tel_from_filename(src)
        except ValueError as exc:
            logging.error("%s: %s", src, exc)
            failures += 1
            continue

        dst = output_path(src, args)
        try:
            frames = stamp_file(src, dst, args.key, tel, force=args.force)
        except (OSError, ValueError) as exc:
            logging.error("%s: %s", src, exc)
            failures += 1
            continue
        logging.info(
            "%s -> %s: %d frames stamped %s=%g", src, dst, frames, args.key, tel
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
