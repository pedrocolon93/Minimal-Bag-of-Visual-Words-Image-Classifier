import math
import shutil

train_percent = 0.8
test_percent = 0.2
import os
from random import shuffle
if __name__ == '__main__':
    corpus_location = "./101_ObjectCategories"
    train_location = "./train"
    test_location = "./test"
    # Create split directories
    if not os.path.exists(train_location):
        os.makedirs(train_location)
    if not os.path.exists(test_location):
        os.makedirs(test_location)

    for (dirpath,dirnames,filenames) in os.walk(corpus_location):
        for category_name in dirnames:
            try:
                os.makedirs(os.path.join(train_location,category_name))
            except Exception as e:
                print(e)
            try:
                os.makedirs(os.path.join(test_location,category_name))
            except Exception as e:
                print(e)
            category_filenames=os.listdir(os.path.join(dirpath, category_name))
            totalfiles = len(category_filenames)
            train_split = int(math.floor(train_percent*totalfiles))
            test_split = int(totalfiles-train_split)
            # category_filenames = [os.path.join(subdirpath,x) for x in category_filenames]
            shuffle(category_filenames)
            train_items = category_filenames[0:train_split]
            test_items = category_filenames[train_split:totalfiles]
            for item in train_items:
                try:
                    src = os.path.join(dirpath,category_name,item)
                    dest = os.path.join(train_location,category_name,item)
                    shutil.move(src,dest)
                except Exception as e:
                    print(e)
            for item in test_items:
                try:
                    shutil.move(os.path.join(dirpath,category_name,item), os.path.join(test_location, category_name, item))
                except Exception as e:
                    print(e)
