#!/usr/bin/env python
# coding: utf-8

# Utils

# In[1]:


from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import pyautogui as p

def plot_image(img, figsize_in_inches=(5,5)):
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()
    
def plot_images(imgs, figsize_in_inches=(5,5)):
    fig, axs = plt.subplots(1, len(imgs), figsize=figsize_in_inches)
    for col, img in enumerate(imgs):
        axs[col].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.show()

def save_image(image: np.ndarray):
        """Save image using opencv"""
        save_file_name = p.prompt(text="Save image name without an extension", title="Enter input image name")
        cv2.imwrite(f"{save_file_name}.png", image)
        sys.stdout.write(f"Image successfully saved as {save_file_name}.png\n")

def obtain_paths_from_dict(raw_range: bool=False):
    """Creates a list of paths of images and returns in ordered format
    ie, [r1_c1, r1_c2, r1_c3.......r2_c1] r:rows & c:cols
    arguments:
    raw_range: if set to true instead of returning .png strings it will return a tuple of (x, y) cordinates, defaultsto fase"""
    x_start = float(CONF_DICT["X_RANGE"][0])
    x_stop = float(CONF_DICT["X_RANGE"][1])
    x_step = float(CONF_DICT["X_RANGE"][2])

    y_start = float(CONF_DICT["Y_RANGE"][0])
    y_stop = float(CONF_DICT["Y_RANGE"][1]) 
    y_step = float(CONF_DICT["Y_RANGE"][2])
    
    x_range_expec = np.arange(x_start, x_stop, x_step)
    y_range_expec = np.arange(y_start, y_stop, y_step)
                   
    if x_step == 0 or x_start == x_stop:
        x_range_expec = [x_start]
        
    if y_step == 0 or y_start == y_stop:
        y_range_expec = [y_start]

    if raw_range:
        return x_range_expec, y_range_expec
    
    paths_expec = []
    for _x in x_range_expec:
        x = f"{_x:.2f}"
        for _y in y_range_expec:
            y = f"{_y:.2f}"
            _path = x + "_" + y + ".png"
            paths_expec.append(_path)
        

    return paths_expec


# Obtain paths

# In[2]:


# Import necessary libraries
import os
import sys
import time
from tkinter import filedialog, messagebox

SCAN_FILE = "scan_details.conf"

class GetConfData:
    """Obtains the config file data present in chosen folder and gets zoom level, X and Y-axes steps.
    The above parameters are then used for Stitching."""

    def __init__(self, parent_folder: str):
        self.folder = parent_folder
        self.folder_contents = os.listdir(self.folder)     # Get folder contents
        self.path_is_valid = self.check_path_validity(self.folder_contents)       # Check if scan_detail.conf exists or not
        self.conf_dict = None

        if not self.path_is_valid:
            # If .conf doesn't exist, throws a warning and makes user choose again
            warning_message = """Select the folder having the following items:
            1) image_folder (having all scaned images)
            2) scan.conf file with .conf extension"""
            messagebox.showwarning(title="Warning", message=warning_message)
            get_folder_name()
        else:
            # If scan_detail.conf exists alongside with images scan_details are extracted
            self.conf_dict = self.return_stitching_dict()

    def check_path_validity(self, contents: list):
        config_files = list()
        for file in contents:
            if file.endswith(".conf"):
                config_files.append(file)
                
        if len(config_files) != 1:
            sys.stdout.write("Here is a list of available scan.config files available in the given folder.\
            Please input the index number of the correct file\n")
            for index, element in enumerate(config_files):
                sys.stdout.write(f"{index}: {element}\n")
            index = input("Enter the index in numeric format: ")
            
            try:
                global SCAN_FILE
                SCAN_FILE = str(config_files[int(index)])
                return True 
            except:
                sys.stderr.write(f"Invalid input please choose from displayed file indexes.\n\n")
                return GetConfData(self.folder)
            
        return True
        

    def return_stitching_dict(self):
        """ Reads through the scan_details.conf and returns a dict of start and end coordinates of scan, its zoom value
        and step_size in each axis and returns a dictionary
        data_dict format is as follows
        key: value --> 'Key': [start_point, end_point, step_size]"""

        scan_path = [path for path in self.folder_contents if SCAN_FILE in path][0]
        scan_path = os.path.join(self.folder, scan_path)
        # Read and dictionary the contents of conf file
        data_dict = dict()
        with open(scan_path, 'r') as f:
            # Opens the conf file and obtains X, Y, Zoom ranges
            conf_content = f.readlines()
            conf_content = [param.replace("\n", "") for param in conf_content if "#" not in param and "FOCUS" not in param]

        data_dict["X_RANGE"] = [ent.split("=")[-1] for ent in conf_content if "X_RANGE" in ent][0].split("|")
        data_dict["Y_RANGE"] = [ent.split("=")[-1] for ent in conf_content if "Y_RANGE" in ent][0].split("|")
        data_dict["ZOOM_RANGE"] = [ent.split("=")[-1] for ent in conf_content if "ZOOM_RANGE" in ent][0].split("|")

        return data_dict


def get_folder_name():
    """Let user select the folder containing images and scan.config file.
    Note: Select the folder having 'image_folder' and scan_detail.conf file"""

    # Returns path of selected folder
    selected_folder_path = filedialog.askdirectory(title="Select Parent Folder containing Images")

    if selected_folder_path == "":
        # If user cancels the select folder path the code ends
        warning_message = "Operation quit by user. Terminating..."
        messagebox.showwarning(title="Warning", message=warning_message)
        sys.exit()

    return selected_folder_path


# In[3]:


folder_path = get_folder_name()
conf_extractor = GetConfData(folder_path)
CONF_DICT = conf_extractor.conf_dict


# Check the integrity of the image folder and file names

# In[4]:


import os
import sys
import shutil
import numpy as np

def rename_validate_files(folder_path: str):
    """This function will rename all files inside the source_folder and save it as x_y coordinates.
    argument:
    folder_path: the path of folder in string format"""
    # Obtains all contents in mentioned path
    files_list = os.listdir(folder_path)
    folder_contents = [(os.path.join(folder_path, file)) for file in files_list]
    is_file_dir = [os.path.isdir(f) for f in folder_contents]
    image_holder_folder_name = folder_contents[is_file_dir.index(True)]

    # Get all image paths by indexing the folder which returns true
    image_paths = os.listdir(image_holder_folder_name)

    # Counts the number of terms in a filename
    terms_in_name = image_paths[0].split('_')
    terms_in_name[-1] = terms_in_name[-1][:-4]

    sys.stdout.write(f"Here is an example filename: {image_paths[0]}\n")
    rename = input("Do you want to rename files? to eg. Xcoor_Ycoor.png [Y or N]:  ").lower()
    if rename not in ["y", "n"]:
        sys.stderr.write(f"Invalid input either input 'y' or 'n'.\n\n")
        return rename_validate_files(folder_path)
        
    if rename == 'n':  # If user doesn't want to rename we do nothing and exit out
        validation = is_folder_ready_for_stitching(image_paths)
        image_paths = [os.path.join(folder_path, image_holder_folder_name,_path) for _path in image_paths]
        return validation, image_paths

    # To maintain homogeneity in file names
    sys.stdout.write(f"Enter the axis index in og name between 1-{len(terms_in_name)}: \n")
    x_index = int(input("X-axis: ")) - 1
    y_index = int(input("Y-axis: ")) - 1

    for k in range(len(image_paths)):
        if '_' in image_paths[k]:  # Splits filenames wrt '_'
            current_path = os.path.join(folder_path, image_holder_folder_name, image_paths[k])
            terms_in_name = image_paths[k].split('_')
            terms_in_name[-1] = terms_in_name[-1][:-4]
            new_name = str(terms_in_name[x_index] + "_" + terms_in_name[y_index] + ".png")
            new_path = os.path.join(folder_path, image_holder_folder_name, new_name)
            shutil.move(current_path, new_path)

    renamed_image_paths = os.listdir(folder_contents[is_file_dir.index(True)])
    sys.stdout.write(f"Files renamed. Example file name: {renamed_image_paths[0]}\n")

    validation = is_folder_ready_for_stitching(renamed_image_paths)
    renamed_image_paths = [os.path.join(folder_path, image_holder_folder_name,_path) for _path in renamed_image_paths]
    return validation, renamed_image_paths
    
    
def is_folder_ready_for_stitching(image_paths: list):
    """This function will check whether there exists an image for every point mentioned in conf file.
    argument:
    image_paths: list of paths of images."""
    
    paths_expec = obtain_paths_from_dict()
    list_image_paths = image_paths[:]
    terms_in_path = len(list_image_paths[0].split("_"))
    if terms_in_path >= 3:
        sys.stderr.write(
            f"""Your given file names is not following the x-coor_y-coor.png format you need to replace in order to"
            prepare the folder\n""")
        sys.stderr.write(f"Your Path isn't Ready For Stitching")

    _extra_paths = []
    for _path in list_image_paths:
        if _path not in paths_expec:
            sys.stderr.write(f"Extra path found --> {_path}\n")
            _extra_paths.append(_path)
    if len(_extra_paths) != 0:
        return False

    errors: int = 0
    filename: str

    # Creates filenames with xcor and ycor lists and checks if they exist in list
    for _path in paths_expec:
        try:
            list_image_paths.remove(_path)
        except ValueError:
            errors += 1
            sys.stderr.write(f"Data Ambiguity found. Expected file not found--> {_path}\n")

    if len(list_image_paths) > 0:
        sys.stdout.write(f"List of \n{list_image_paths}\n")
        errors += len(list_image_paths)

    if errors == 0:
        sys.stdout.write(f"Dataset verified you may go ahead\n")
        return True
    else:
        return False


# In[5]:


validation, filepaths_list = rename_validate_files(folder_path)
validation, filepaths_list


# Resize images

# In[6]:


import cv2 as cv

def resize_images(imgs, size_factor=0.5):
    """This function resizes all input images within themselves.
    arguments:
    imgs: Either a single ndarray or list of ndarray.
    size_factor: Must be a float between 0 to 1. By what perfect an image has to be resized. Defaults to 50% ie. 0.5
    No returns."""
    if isinstance(imgs, list):
        for idx in range(len(imgs)):
            imgs[idx] = cv.resize(imgs[idx], None, fx=size_factor, fy=size_factor, interpolation=cv.INTER_LINEAR_EXACT)
    else:
        imgs = cv.resize(imgs, None, fx=size_factor, fy=size_factor, interpolation=cv.INTER_LINEAR_EXACT)


# In[67]:


image_data_list = [cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB) for filename in filepaths_list]

RESIZE_RATIO = 0.5
resize_images(image_data_list, size_factor=RESIZE_RATIO)    # Resized image to self be careful how u run it it overwrites the existing list


# In[68]:


np.shape(image_data_list[-2])


# In[ ]:


from stitching.images import Images

images = Images.of(weir_imgs)

medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
low_imgs = list(images.resize(Images.Resolution.LOW))
final_imgs = list(images.resize(Images.Resolution.FINAL))


# In[ ]:


plot_images(low_imgs, (20,20))


# In[ ]:


original_size = images.sizes[0]
medium_size = images.get_image_size(medium_imgs[0])
low_size = images.get_image_size(low_imgs[0])
final_size = images.get_image_size(final_imgs[0])

print(f"Original Size: {original_size}  -> {'{:,}'.format(np.prod(original_size))} px ~ 1 MP")
print(f"Medium Size:   {medium_size}  -> {'{:,}'.format(np.prod(medium_size))} px ~ 0.6 MP")
print(f"Low Size:      {low_size}   -> {'{:,}'.format(np.prod(low_size))} px ~ 0.1 MP")
print(f"Final Size:    {final_size}  -> {'{:,}'.format(np.prod(final_size))} px ~ 1 MP")


# Feature Detector

# In[69]:


from stitching.feature_detector import FeatureDetector

finder = FeatureDetector(detector="orb",
                         nfeatures=1000,
                         scaleFactor=1.2,
                         edgeThreshold=10,
                         firstLevel=1,
                         nlevels=10,
                         WTA_K=4,
                         patchSize=10,
                         fastThreshold=10,
                        )

features = [finder.detect_features(img) for img in image_data_list]
# features = finder.detect_features(image_data_list[0][: , -240:])
# keypoints = features.getKeypoints()
# new_keypoints = []
# drawn_op = finder.draw_keypoints(image_data_list[0], features)


# In[70]:


for feature in features:
    keypoints =  feature.getKeypoints()
    print(len(keypoints))
    
    for kp in keypoints:
        print(kp.pt)
    print()


# Match Features

# In[71]:


from stitching.feature_matcher import FeatureMatcher

"""
We can look at the confidences, which are calculated by:
`confidence = number of inliers / (8 + 0.3 * number of matches)` 
(Lines 435-7 of [this file](https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/stitching/src/matchers.cpp))

The inliers are calculated using the random sample consensus (RANSAC) method, e.g. 
in [this file](https://github.com/opencv/opencv/blob/68d15fc62edad980f1ffa15ee478438335f39cc3/modules/stitching/src/matchers.cpp) in Line 425. 
We can plot the inliers which is shown later.
"""

matcher = FeatureMatcher(matcher_type='homography', range_width=-1)
matches = matcher.match_features(features)


# In[72]:


matcher.get_confidence_matrix(matches)


# In[73]:


all_relevant_matches = matcher.draw_matches_matrix(image_data_list, features, matches, conf_thresh=1, 
                                                   inliers=True, matchColor=(0, 255, 0))

for idx1, idx2, img in all_relevant_matches:
    print(f"Matches Image {idx1+1} to Image {idx2+1}")
    plot_image(img, (10,5))


# Subsetter

# In[74]:


from stitching.subsetter import Subsetter

subsetter = Subsetter()
dot_notation = subsetter.get_matches_graph(filepaths_list, matches)
print(dot_notation)


def subset(image_path, image_data_list, indices):
    subset_paths, subset_images = None, None
    subset_paths = [image_path[i] for i in indices]
    subset_images = [image_data_list[i] for i in indices]
    return subset_paths, subset_images


# In[75]:


indices = subsetter.get_indices_to_keep(features, matches)

medium_imgs = subsetter.subset_list(image_data_list, indices)
features = subsetter.subset_list(features, indices)
matches = subsetter.subset_matches(matches, indices)

subset_paths, subset_images = subset(filepaths_list, image_data_list, indices)

print(subset_paths)
print()
print(matcher.get_confidence_matrix(matches))


# Camera Estimation, Adjustion and Correction

# In[76]:


from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector

camera_estimator = CameraEstimator(estimator='homography')
camera_adjuster = CameraAdjuster(adjuster='ray', refinement_mask='xxxxx')
wave_corrector = WaveCorrector(wave_correct_kind='horiz')    # horiz, vert, auto, no

cameras = camera_estimator.estimate(features, matches)
cameras = camera_adjuster.adjust(features, matches, cameras)
cameras = wave_corrector.correct(cameras)


# Warp Images

# In[77]:


from stitching.warper import Warper

warper = Warper(warper_type='spherical')
warper.set_scale(cameras)


# In[85]:


warped_images = list(warper.warp_images(image_data_list, cameras, RESIZE_RATIO))

image_sizes = [list(np.shape(img))[:2][::-1] for img in image_data_list]
warped_mask = list(warper.create_and_warp_masks(image_sizes, cameras, RESIZE_RATIO))
corners, sizes = warper.warp_rois(image_sizes, cameras, RESIZE_RATIO)


# In[86]:


plot_images(warped_images, (40,20))
plot_images(warped_mask, (40,20))


# In[87]:


print(corners)
print(sizes)


# Timelapser

# In[89]:


from stitching.timelapser import Timelapser

timelapser = Timelapser('as_is')  # types = no, crop
timelapser.initialize(final_corners, final_sizes)

for img, corner in zip(warped_images, final_corners):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    plot_image(frame, (5,10))


# Crop

# In[100]:


from stitching.cropper import Cropper

"""without black borders, estimate the largest joint interior rectangle. `Cropper(crop=True)`"""
cropper = Cropper(crop=True)


# In[92]:


panaroma_mask = cropper.estimate_panorama_mask(warped_images, warped_mask, corners, sizes)
plot_image(mask, (5,5))


# In[101]:


# The estimation of the largest interior rectangle is not yet implemented in OpenCV, 
# but a [Numba](https://numba.pydata.org/) Implementation by my own. 
# You check out the details [here](https://github.com/lukasalexanderweber/lir). 
# Compiling the Code takes a bit (only once, the compiled code is then cached)
# (https://numba.pydata.org/numba-doc/latest/developer/caching.html) for future function calls)


# In[94]:


largest_interior_rectangle = cropper.estimate_largest_interior_rectangle(panaroma_mask)
print(largest_interior_rectangle)


# In[105]:


plot = largest_interior_rectangle.draw_on(panaroma_mask, size=2)
plot_image(plot, (5,5))


# In[106]:


corners_zeroed = cropper.get_zero_center_corners(corners)
rectangles = cropper.get_rectangles(corners_zeroed, sizes)

plot = rectangles[5].draw_on(plot, (0, 255, 0), 2)  # The rectangle of the center img
plot_image(plot, (5,5))


# In[107]:


overlap = cropper.get_overlap(rectangles[1], largest_interior_rectangle)
plot = overlap.draw_on(plot, (255, 0, 0), 2)
plot_image(plot, (5,5))


# In[108]:


intersection = cropper.get_intersection(rectangles[1], overlap)
plot = intersection.draw_on(warped_mask[1], (255, 0, 0), 2)
plot_image(plot, (2.5,2.5))


# In[122]:


cropper.prepare(warped_images, warped_mask, corners, sizes)

cropped_imgs = list(cropper.crop_images(warped_images))
cropped_masks = list(cropper.crop_images(warped_mask))
cropped_corners, cropped_sizes = cropper.crop_rois(corners, sizes)


# In[123]:


timelapser = Timelapser('as_is')
timelapser.initialize(cropped_corners, low_sizes)

for img, corner in zip(cropped_imgs, cropped_corners):
    timelapser.process_frame(img, corner)
    frame = timelapser.get_frame()
    plot_image(frame, (10,10))


# Now we need stategies how to compose the already overlaying images into one panorama image. Strategies are:
# 
# - Seam Masks
# - Exposure Error Compensation
# - Blending

# Seam Masks

# In[1]:


from stitching.seam_finder import SeamFinder

"""
Seam masks find a transition line between images with the least amount of interference
"""

seam_finder = SeamFinder(finder='dp_color')

seam_masks = seam_finder.find(cropped_imgs, cropped_corners, cropped_masks)
seam_masks = [seam_finder.resize(seam_mask, mask) for seam_mask, mask in zip(seam_masks, cropped_masks)]

seam_masks_plots = [SeamFinder.draw_seam_mask(img, seam_mask) for img, seam_mask in zip(cropped_imgs, seam_masks)]
plot_images(seam_masks_plots, (15,10))


# Exposure Error Compensation

# In[125]:


from stitching.exposure_error_compensator import ExposureErrorCompensator

"""
Frequently exposure errors respectively exposure differences between images occur which lead to artefacts in the final panorama.

"""
# compensator = 'gain_blocks', 'gain', 'channel', 'channel_blocks', 'no'
compensator = ExposureErrorCompensator(compensator='gain_blocks', nr_feeds=1, block_size=32)    
compensator.feed(corners, cropped_corners, cropped_masks)
compensated_imgs = [compensator.apply(idx, corner, img, mask) 
                    for idx, (img, mask, corner) 
                    in enumerate(zip(cropped_imgs, cropped_masks, cropped_corners))]

plot_images(compensated_imgs, (20, 10))


# Blending

# In[146]:


from stitching.blender import Blender

"""
blender_type: "multiband", "feather", "no"
blend_strength: int
"""

blender = Blender(blender_type='no', blend_strength=20)
blender.prepare(cropped_corners, cropped_sizes)
for img, mask, corner in zip(compensated_imgs, seam_masks, cropped_corners):
    blender.feed(img, mask, corner)
panorama, _ = blender.blend()


# In[147]:


plot_image(panorama, (20,20))


# In[150]:


blended_seam_masks = seam_finder.blend_seam_masks(seam_masks, cropped_corners, cropped_sizes)

plot_image(seam_finder.draw_seam_lines(panorama, blended_seam_masks, linesize=3), (15,10))
plot_image(seam_finder.draw_seam_polygons(panorama, blended_seam_masks), (15,10))

