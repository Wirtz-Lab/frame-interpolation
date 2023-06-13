import os,pathlib,shutil

src = r'\\fatherserverdw\Saurabh\Saurabh\Pancreas_Ashley_Files\training_tiles_triplet_v2\sequences'
srcfn = os.path.basename(src)
target = [26,27,28,32,50,55,78]
move to new folder

dst = r'\\fatherserverdw\Saurabh\Saurabh\Pancreas_Ashley_Files\training_tiles_triplet_v2_test\sequences'
if not os.path.exists(dst) os.mkdir(dst)