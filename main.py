# Import program files
import matplotlib.pyplot as plt
import numpy as np
import sys
from MyUtils import find_thresh_conf_matrix
from PathSelector import *
from CheckFiles import *
from Beautifiers import *
from stitching.feature_detector import FeatureDetector
# Libraries for multiprocessing/multithreading
from threading import Thread, Event
from time import time
import cv2 as cv
import torch

# CONSTANTS
RESIZE_RATIO = 1  # Images will be resized to RESIZED_RATIO
# FINAL_RESIZE_RATIO = 1
BATCH_SIZE = 8  # How many images to stitch at a time
try_use_gpu = torch.cuda.is_available()

sys.stdout.write(f"Stitching to generate panorama is a 9-stage process.\n\n")
# Get folder path
folder_path = get_folder_name()
# Obtains the config file details
conf_extractor = GetConfData(folder_path)
# The config file details
CONF_DICT = conf_extractor.conf_dict
# Validate the image folder paths and returns the status
validation, filepaths_list = rename_validate_files(CONF_DICT, folder_path)
if not validation:
    sys.stderr.write(f"\nYour image folder isn't what the stitcher requires. Exiting...")
    sys.exit()
# Obtains the image data from paths but appends in order of file name asc to dsc
start = time()
image_data_list = [cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB) for filename in filepaths_list]
# Resizes images for better stitching
resize_images(image_data_list, size_factor=RESIZE_RATIO)

ROW_PANO = {}
x_range, y_range = obtain_paths_from_dict(CONF_DICT, raw_range=True)


def stitch_rows(y_coor, range_x, path_list):
    sys.stdout.write(f"Stitching thread for stitching for for y_coordinate {y_coor} has started... \n\n")
    # Get paths
    paths = [full_path for full_path in path_list if ('_'.join([str(x), str(y_coor)]) for x in range_x)]
    # Get image_datas
    images = [cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB) for path in paths]
    sys.stdout.write(f"STAGE 1: Images acquired.\n")

    panorama = stitch_images(images)
    sys.stdout.write(f"STAGE 9: Row panorama Image generated for {y_coor}.\n\n")

    image_name = f"row_{y_coor}.png"
    print(image_name)

    cv.imwrite(image_name, panorama)
    sys.stdout.write(f"Row-Panorama saved as {image_name}")

    global ROW_PANO
    ROW_PANO[str(y_coor)] = panorama


def stitch_images(images: list, nfeatures=500, thread_no=None):
    # Instantiate BeautifierFunctions class
    beautifier = BeautifierFunctions()
    if thread_no is None:
        thread_no = ""

    orb_detector = cv.ORB.create(nfeatures=nfeatures,
                                 scaleFactor=1.1,
                                 edgeThreshold=0,
                                 # firstLevel=1,
                                 # nlevels=10,
                                 WTA_K=2,
                                 patchSize=10,
                                 fastThreshold=10,
                                 )
    """
    :BestOf2NearestMatcher	(	bool 	try_use_gpu = false,
                                float 	match_conf = 0.3f,
                                int 	num_matches_thresh1 = 6,
                                int 	num_matches_thresh2 = 6,
                                double 	matches_confindece_thresh = 3.
                                )	
    
    try_use_gpu	                Should try to use GPU or not
    match_conf	                Match distances ration threshold
    num_matches_thresh1	        Minimum number of matches required for the 2D projective transform estimation used in the
                                inliers classification step
    num_matches_thresh2	        Minimum number of matches required for the 2D projective transform re-estimation on inliers
    matches_confindece_thresh	Matching confidence threshold to take the match into account. 
                                The threshold was determined experimentally and set to 3 by default.       
    """
    # match_conf = 0.35
    # # if thread_no is None:
    # #     match_conf = 0.2
    # matcher = cv.detail_BestOf2NearestMatcher(try_use_gpu=try_use_gpu,
    #                                           match_conf=match_conf,
    #                                           num_matches_thresh1=6,  #int(nfeatures*0.10),
    #                                           num_matches_thresh2=6,  #int(nfeatures*0.10),
    #                                           matches_confindece_thresh=3,
    #                                           )

    # Find features for all images available
    features = [cv.detail.computeImageFeatures2(orb_detector, img) for img in images]
    # features = [feature_detector.detect_features(img) for img in images]
    sys.stdout.write(f"STAGE 2_{thread_no}: Features obtained.\n")
    # Find matches in features
    matches = beautifier.matcher.match_features(features)
    sys.stdout.write(f"STAGE 3_{thread_no}: Matches found.\n")
    # Obtains the confidence matrix from matches
    conf_matrix = get_confidence_matrix(matches)
    # Null all matcher other than adjacent
    for row in range(len(conf_matrix)):
        for d_id in range(len(conf_matrix[row])):
            conf_matrix[row][d_id] = conf_matrix[row][d_id] if d_id in range(row - 1, row + 2) else 0
    # Calculates the minimum threshold from matrix
    conf_matrix_threshold = find_thresh_conf_matrix(conf_matrix)
    sys.stdout.write(f"\t\t CONF_THRESHOLD calculated:  {conf_matrix_threshold}\n")
    # Finds normalised camera details for each image
    cameras = beautifier.obtain_cameras(features, matches, conf_matrix_threshold)
    sys.stdout.write(f"STAGE 4_{thread_no}: Cameras oriented.\n")
    # Here we get the warped images
    warped_images, warped_masks, corners, sizes = beautifier.obtain_warper_outputs(cameras, images, RESIZE_RATIO)
    sys.stdout.write(f"STAGE 5_{thread_no}: Warping finished.\n")
    # Since post warping there exists black background at places we crop out the background
    cropped_images, cropped_masks, cropped_corners, cropped_sizes = beautifier.obtain_cropper_outputs(warped_images,
                                                                                                      warped_masks,
                                                                                                      corners, sizes)
    sys.stdout.write(f"STAGE 6_{thread_no}: Cropping done to remove black background.\n")
    # Seam masks are the portion from each image which are used to form the panorama image
    seam_masks = beautifier.obtain_seam_masks(cropped_images, cropped_masks, cropped_corners)
    sys.stdout.write(f"STAGE 7_{thread_no}: Seam-Masks obtained.\n")
    # Exposure of images is changed to merge all of them together
    compensated_images = beautifier.exposure_correction(cropped_images, cropped_masks, cropped_corners, corners)
    sys.stdout.write(f"STAGE 8_{thread_no}: Exposure Correction done.\n")
    # Here is out final panorama image and its status whether it's stitched or not
    panorama, status = beautifier.blend_images(compensated_images, seam_masks, cropped_corners, cropped_sizes)
    return panorama


PIXEL_TO_MM = get_mm2pixel_map(float(conf_extractor.conf_dict["ZOOM_RANGE"][0]))
OVERLAP_VARY_ERROR_PERCENTAGE = 10

X_OVERLAP_POINT_REGION = calculate_overlap_start_point(pixel_to_mm=PIXEL_TO_MM,
                                                       step_size=float(conf_extractor.conf_dict["X_RANGE"][-1]),
                                                       error_percentage=OVERLAP_VARY_ERROR_PERCENTAGE)

total_images = len(x_range) * len(y_range)
if BATCH_SIZE > total_images / 2:
    BATCH_SIZE = total_images
num_epochs = math.ceil(total_images / BATCH_SIZE)
row_epoch_images = {}

batch_images_path = "stitched_batch"
if not os.path.exists(batch_images_path):
    os.mkdir(batch_images_path)
else:
    for file_path in os.listdir(batch_images_path):
        os.remove(os.path.join(batch_images_path, file_path))

if not os.path.exists(batch_images_path):
    os.mkdir(batch_images_path)


def stitch_batches(epoch):
    sys.stdout.write(f"Thread for epoch:{epoch} started.\n")
    filepaths_list_len = len(filepaths_list)
    path_list = filepaths_list[epoch*BATCH_SIZE: min((epoch+1)*BATCH_SIZE, filepaths_list_len)]
    images_list = [
        cv.resize(cv.cvtColor(cv.imread(path), cv.COLOR_BGR2RGB), None, fx=RESIZE_RATIO, fy=RESIZE_RATIO) for path in
        path_list
    ]
    while True:
        # Runs loop for stitching and saving rows if error occurs then restarts the thread
        try:
            row_epoch_images[epoch] = stitch_images(images_list, thread_no=epoch)
            row_epoch_images[epoch] = np.array(row_epoch_images[epoch], dtype=np.int16)
            break
        except Exception as e:
            sys.stderr.write(f"Error {e} while stitching thread:{epoch}, restarting stitching...\n")

    cv.imwrite(f"{batch_images_path}/{epoch}.png", row_epoch_images[epoch])
    sys.stdout.write(f"Thread for epoch:{epoch} ended.\n\n")


row_stitching_threads = {}
for epoch in range(num_epochs):
    row_stitching_threads[epoch] = Thread(target=stitch_batches, args=(epoch,))
    row_stitching_threads[epoch].start()

for thread in list(row_stitching_threads.values()):
    thread.join()

batch_images = []
for path in sorted(os.listdir(batch_images_path)):
    _image = cv.cvtColor(cv.imread(os.path.join(batch_images_path, path)), cv.COLOR_BGR2RGB)
    batch_images.append(_image)

sys.stdout.write(f"Batch Image Stitching Starts here...\n")

while True:
    try:
        final_panorama = None
        if len(batch_images) != 1:
            final_panorama = stitch_images(batch_images, nfeatures=np.shape(batch_images[0])[1], thread_no=None)
        else:
            final_panorama = batch_images[0]
        end = time()
        sys.stdout.write(f"Stitching took {end - start} seconds\n")
        plt.imshow(final_panorama)
        plt.show()
        # Saving out output image as `output.png`
        cv.imwrite(f"{batch_images_path}/panorama.png", final_panorama)
        break
    except Exception as e:
        sys.stderr.write(f"{e}")