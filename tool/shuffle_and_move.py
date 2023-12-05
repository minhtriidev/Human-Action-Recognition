import os
import random
import shutil

src_dir = "D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/JUMP"
train_dir = "D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/train/jump"
valid_dir = "D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/valid/jump"
test_dir = "D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/test/jump"
sub_dir = "D:/Documents/aHCMUT_Documents/DATN/BLENDER/TRAIN_VAL_TEST_6/sub/jump"

# Get only video files with ".mp4" extension
video_files = [file for file in os.listdir(src_dir) if file.lower().endswith(".mp4")]
random.shuffle(video_files)

num_train = 245
num_valid = 70
num_test = 35

for i, video in enumerate(video_files):
    src_path = os.path.join(src_dir, video)
    if i < num_train:
        dst_path = os.path.join(train_dir, video)
    elif i < num_train + num_valid:
        dst_path = os.path.join(valid_dir, video)
    else:
        # Check if the test directory already has 30 files, if not, move the file to "test" directory
        test_files_count = len(os.listdir(test_dir))
        if test_files_count < num_test:
            dst_path = os.path.join(test_dir, video)
        else:
            dst_path = os.path.join(sub_dir, video)
    shutil.copy(src_path, dst_path)
