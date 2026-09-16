import os
import random

# define the directories for training, testing, and validation sets
train_dir = './data/x2ka_aircas/paired/train'
test_dir = './data/x2ka_aircas/paired/test/'
val_dir = './data/x2ka_aircas/paired/val/'

# set the random seed for reproducibility
random.seed(42)
tiff_dir = './data/aircas/paired'
files = os.listdir(tiff_dir)

# randomly shuffle the list of image files
random.shuffle(files)

# split the list of image files into training, testing, and validation sets
n = len(files)
n_train = int(0.7 * n)
n_test = int(0.2 * n)
n_val = n - n_train - n_test

train_files = files[:n_train]
test_files = files[n_train:n_train+n_test]
val_files = files[n_train+n_test:]

# create symbolic links for the training set
for filename in train_files:
    src = os.path.join(tiff_dir, filename)
    dst = os.path.join(train_dir, filename)
    os.symlink(os.path.abspath(src), os.path.abspath(dst))

# create symbolic links for the testing set
for filename in test_files:
    src = os.path.join(tiff_dir, filename)
    dst = os.path.join(test_dir, filename)
    os.symlink(os.path.abspath(src), os.path.abspath(dst))

# create symbolic links for the validation set
for filename in val_files:
    src = os.path.join(tiff_dir, filename)
    dst = os.path.join(val_dir, filename)
    os.symlink(os.path.abspath(src), os.path.abspath(dst))
