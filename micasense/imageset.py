#!/usr/bin/env python
# coding: utf-8
"""
RedEdge Capture Class

    A Capture is a set of images taken by one RedEdge cameras which share
    the same unique capture identifier.  Generally these images will be
    found in the same folder and also share the same filename prefix, such
    as IMG_0000_*.tif, but this is not required

Copyright 2017 MicaSense, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in the
Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import fnmatch
import os
from typing import Callable, List, Optional, Tuple, Any

from micasense.mp_config import spawn_pool

import exiftool

import micasense.capture as capture
import micasense.image as image
from micasense.imageutils import save_capture as save_capture


def image_from_file(filename: str) -> image.Image:
    return image.Image(filename)


def _align_single_cap(args: Tuple[capture.Capture, Optional[List]]) -> capture.Capture:
    cap, warp_matrices = args
    cap.create_aligned_capture(warp_matrices=warp_matrices)
    return cap


class ImageSet:
    """
    An ImageSet is a container for a group of captures that are processed together
    """

    def __init__(self, captures: List[capture.Capture]):
        self.captures = captures
        captures.sort()

    @classmethod
    def from_directory(
        cls,
        directory: str,
        progress_callback: Optional[Callable[[float], None]] = None,
        allow_uncalibrated: bool = False,
    ) -> "ImageSet":
        """
        Create and ImageSet recursively from the files in a directory
        """
        cls.basedir = directory
        matches = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, "*.tif"):
                matches.append(os.path.join(root, filename))

        images = []

        with exiftool.ExifToolHelper() as exift:
            for i, path in enumerate(matches):
                images.append(
                    image.Image(
                        path, exiftool_obj=exift, allow_uncalibrated=allow_uncalibrated
                    )
                )
                if progress_callback is not None:
                    progress_callback(float(i) / float(len(matches)))

        # create a dictionary to index the images, so we can sort them
        # into captures
        # {
        #     "capture_id": [img1, img2, ...]
        # }
        captures_index = {}
        for img in images:
            c = captures_index.get(img.capture_id)
            if c is not None:
                c.append(img)
            else:
                captures_index[img.capture_id] = [img]
        captures = []
        for cap_imgs in captures_index:
            imgs = captures_index[cap_imgs]
            newcap = capture.Capture(imgs)
            captures.append(newcap)
        if progress_callback is not None:
            progress_callback(1.0)
        return cls(captures)

    def as_nested_lists(self) -> Tuple[List[List[Any]], List[str]]:
        """
        Get timestamp, latitude, longitude, altitude, capture_id, dls-yaw, dls-pitch, dls-roll, and irradiance from all
        Captures.
        :return: List data from all Captures, List column headers.
        """
        columns = [
            "timestamp",
            "latitude",
            "longitude",
            "altitude",
            "capture_id",
            "dls-yaw",
            "dls-pitch",
            "dls-roll",
            "file_paths",
        ]
        irr = ["irr-{}".format(wve) for wve in self.captures[0].center_wavelengths()]
        columns += irr
        data = []
        for cap in self.captures:
            dat = cap.utc_time()
            loc = list(cap.location())
            uuid = cap.uuid
            dls_pose = list(cap.dls_pose())
            irr = cap.dls_irradiance()
            paths = [img.path for img in cap.images]
            row = [dat] + loc + [uuid] + dls_pose + [paths] + irr
            data.append(row)
        return data, columns

    def dls_irradiance(self) -> dict:
        """
        Get utc_time and irradiance for each Capture in ImageSet.
        :return: dict {utc_time : [irradiance, ...]}
        """
        series = {}
        for cap in self.captures:
            dat = cap.utc_time().isoformat()
            irr = cap.dls_irradiance()
            series[dat] = irr
        return series

    def align_captures(
        self,
        warp_matrices: Optional[List] = None,
        multiprocess: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        """
        Align images for all Captures in the ImageSet in parallel or serially.
        """
        if multiprocess:
            with spawn_pool() as pool:
                aligned_caps = []
                args_list = [(cap, warp_matrices) for cap in self.captures]
                for i, cap in enumerate(pool.imap(_align_single_cap, args_list)):
                    aligned_caps.append(cap)
                    if progress_callback is not None:
                        progress_callback(float(i + 1) / float(len(self.captures)))
                self.captures = aligned_caps
        else:
            for i, cap in enumerate(self.captures):
                _align_single_cap((cap, warp_matrices))
                if progress_callback is not None:
                    progress_callback(float(i + 1) / float(len(self.captures)))

    def save_stacks(
        self,
        warp_matrices: List,
        stack_directory: str,
        thumbnail_directory: Optional[str] = None,
        irradiance: Optional[List[float]] = None,
        multiprocess: bool = True,
        overwrite: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        if not os.path.exists(stack_directory):
            os.makedirs(stack_directory)
        if thumbnail_directory is not None and not os.path.exists(thumbnail_directory):
            os.makedirs(thumbnail_directory)

        save_params_list = []
        for local_capture in self.captures:
            save_params_list.append(
                {
                    "output_path": stack_directory,
                    "thumbnail_path": thumbnail_directory,
                    "file_list": [img.path for img in local_capture.images],
                    "warp_matrices": warp_matrices,
                    "irradiance_list": irradiance,
                    "photometric": "MINISBLACK",
                    "overwrite_existing": overwrite,
                }
            )

        if multiprocess:
            with spawn_pool() as pool:
                for i, _ in enumerate(
                    pool.imap_unordered(save_capture, save_params_list)
                ):
                    if progress_callback is not None:
                        progress_callback(float(i) / float(len(save_params_list)))
        else:
            for params in save_params_list:
                save_capture(params)

    def __repr__(self):
        return f"ImageSet(num_captures={len(self.captures)})"
