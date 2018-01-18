import _pickle

# import libsvm
# import argparse
# from learn import extractSift, computeHistograms, writeHistogramsToFile
from sklearn import preprocessing

from plain_python_learn import extractSift,computeHistograms,stackHistogramData
import cv2 as cv

from plain_python_learn import stackHistogramData

HISTOGRAMS_FILE = 'testdata.svm'
CODEBOOK_FILE = 'codebook.file'
MODEL_FILE = 'trainingdata.svm.model'

# def extractSift(input_files):
#     print("extracting Sift features")
#     all_features_dict = {}
#     for i, fname in enumerate(input_files):
#         features_fname = fname + '.sift'
#         print("calculating sift features for", fname)
#         # sift.process_image(fname, features_fname)
#         # print "gathering sift features for", fname,
#         # locs, descriptors = sift.read_features_from_file(features_fname)
#         img1 = cv.imread(fname, 0)  # queryImage
#         # Initiate SIFT detector
#         # orb = cv.ORB_create(150)
#         # t = cv.xfeatures2d.SURF_create()
#         orb = cv.xfeatures2d.SIFT_create(150)
#         # Compute keypoints
#         locs, descriptors = orb.detectAndCompute(img1, None)
#         print(descriptors.shape)
#         all_features_dict[fname] = descriptors
#     return all_features_dict


print("---------------------")
print("## extract Sift features")
all_files = []
all_files_labels = {}
all_features = {}

# args = parse_arguments()
model_file = "/mnt/hgfs/PycharmProjects/objectdetectioncamera/corpus/train/trainingdata.svm.model"
codebook_file = "/mnt/hgfs/PycharmProjects/objectdetectioncamera/corpus/train/codebook.file"

fnames = ["/home/pedro/Documents/101traintest/test/accordion/image_0008.jpg",
"/home/pedro/Documents/101traintest/test/accordion/image_0011.jpg",
"/home/pedro/Documents/101traintest/test/accordion/image_0020.jpg",
"/home/pedro/Documents/101traintest/test/accordion/image_0045.jpg",
"/home/pedro/Documents/101traintest/test/accordion/image_0051.jpg",
"/home/pedro/Documents/101traintest/test/accordion/image_0052.jpg"]
# fnames = ["/home/pedro/Documents/101traintest/test/yin_yang/image_0006.jpg"]
all_features = extractSift(fnames)
for i in fnames:
    all_files_labels[i] = 0  # label is unknown

print("---------------------")
print("## loading codebook from " + codebook_file)
with open(codebook_file, 'rb') as f:
    codebook = _pickle.load(f,encoding="latin1")

print("---------------------")
print("## computing visual word histograms")
all_word_histgrams = {}
for imagefname in all_features:
    word_histgram = computeHistograms(codebook, all_features[imagefname])
    all_word_histgrams[imagefname] = word_histgram

print("---------------------")
print("## write the histograms to file to pass it to the svm")
nclusters = codebook.shape[0]
# writeHistogramsToFile(nclusters,
#                       all_files_labels,
#                       fnames,
#                       all_word_histgrams,
#                       HISTOGRAMS_FILE)
X = stackHistogramData(nclusters,
                           all_files_labels,
                           fnames,
                           all_word_histgrams)
print("---------------------")
print("## test data with svm")
model_file = open("scikit_svm.model","rb")
svm_model = _pickle.load(model_file)
X = X[:,1:]
X_scaled = preprocessing.scale(X)
print(svm_model.predict(X))
# print(libsvm.test(HISTOGRAMS_FILE, model_file))
