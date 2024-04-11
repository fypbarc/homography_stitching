# Import necessary libraries
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.cropper import Cropper
from stitching.feature_matcher import FeatureMatcher

import sys
import numpy as np
from MyUtils import *
import math

from stitching.seam_finder import SeamFinder
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.blender import Blender
import cv2 as cv

from threading import Thread
import time


ADJUSTOR_BATCH_SIZE = 5
CAMERAS = []


class BeautifierFunctions:

    def __init__(self):
        self.matcher = FeatureMatcher(matcher_type="homography", range_width=-1)
        self.camera_estimator = cv.detail_HomographyBasedEstimator()
        self.camera_adjuster = cv.detail_BundleAdjusterRay()
        self.refinement_mask = np.array([[1, 1, 1],
                                         [0, 1, 1],
                                         [0, 0, 0]], np.uint8)
        self.confidence_threshold = None
        self.wave_corrector = cv.detail.WAVE_CORRECT_HORIZ  # horiz, vert, auto, no
        self.warper = Warper(warper_type='spherical')
        self.cropper = Cropper(crop=True)
        self.seam_finder = SeamFinder(finder='voronoi')
        self.compensator = ExposureErrorCompensator(compensator='gain_blocks', nr_feeds=1, block_size=32)
        self.blender = Blender(blender_type='multiband', blend_strength=15)
        sys.stdout.write("BeautifierFunctions setup finished...\n")

    def estimator_estimate(self, features, pairwise_matches):
        b, cameras = self.camera_estimator.apply(features, pairwise_matches, None)
        if not b:
            raise Exception("Homography estimation failed.")
        for cam in cameras:
            cam.R = cam.R.astype(np.float32)
        return cameras

    def wave_corrector_correct(self, cameras):
        if self.wave_corrector is not None:
            rmats = [np.copy(cam.R) for cam in cameras]
            rmats = cv.detail.waveCorrect(rmats, self.wave_corrector)
            for idx, cam in enumerate(cameras):
                cam.R = rmats[idx]
            return cameras
        return cameras

    def adjustor_adjust(self, confidence_threshold, features, pairwise_matches, estimated_cameras):
        self.camera_adjuster.setConfThresh(confidence_threshold)
        self.camera_adjuster.setRefinementMask(self.refinement_mask)

        b, cameras = self.camera_adjuster.apply(features, pairwise_matches, estimated_cameras)
        if not b:
            raise Exception("Error while adjusting cameras")
        return cameras

    def obtain_cameras(self, features, matches, confidence_threshold):
        try:
            cameras = self.estimator_estimate(features, matches)
            cameras = self.adjustor_adjust(confidence_threshold, features, matches, cameras)
            cameras = self.wave_corrector_correct(cameras)
            if cameras is None:
                raise Exception("Couldn't calculate for cameras")
            return cameras
        except Exception as e:
            sys.stderr.write(f"{e}")

    def obtain_warper_outputs(self, cameras, image_data_list, resize_ratio):
        try:
            self.warper.set_scale(cameras)
            warped_images = list(self.warper.warp_images(image_data_list, cameras, resize_ratio))
            image_sizes = [list(np.shape(img))[:2][::-1] for img in image_data_list]
            warped_mask = list(self.warper.create_and_warp_masks(image_sizes, cameras, resize_ratio))
            corners, sizes = self.warper.warp_rois(image_sizes, cameras, resize_ratio)
            return warped_images, warped_mask, corners, sizes
        except Exception as e:
            sys.stdout.write(f"Couldn't set warper scale as the following Exception occurred\n{e}\n")

    def obtain_cropper_outputs(self, warped_images, warped_mask, corners, sizes):
        self.cropper.prepare(warped_images, warped_mask, corners, sizes)
        cropped_images = list(self.cropper.crop_images(warped_images))
        cropped_masks = list(self.cropper.crop_images(warped_mask))
        cropped_corners, cropped_sizes = self.cropper.crop_rois(corners, sizes)
        return cropped_images, cropped_masks, cropped_corners, cropped_sizes

    def obtain_seam_masks(self, cropped_images, cropped_masks, cropped_corners):
        seam_masks = self.seam_finder.find(cropped_images, cropped_corners, cropped_masks)
        seam_masks = [self.seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_masks)]
        return seam_masks

    def exposure_correction(self, cropped_images, cropped_masks, cropped_corners, corners):
        self.compensator.feed(corners, cropped_corners, cropped_masks)
        compensated_images = [self.compensator.apply(idx, corner, img, mask)
                              for idx, (img, mask, corner)
                              in enumerate(zip(cropped_images, cropped_masks, cropped_corners))]
        return compensated_images

    def blend_images(self, compensated_images, seam_masks, cropped_corners, cropped_sizes):
        self.blender.prepare(cropped_corners, cropped_sizes)
        for img, mask, corner in zip(compensated_images, seam_masks, cropped_corners):
            self.blender.feed(img, mask, corner)
        panorama, _ = self.blender.blend()
        return panorama, _
