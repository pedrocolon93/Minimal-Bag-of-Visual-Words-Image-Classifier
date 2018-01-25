from os.path import exists, isdir, basename, join, splitext
import pickle
import _pickle
import scipy
import scipy.stats

from glob import glob
from numpy import zeros, resize, sqrt, histogram, hstack, vstack, savetxt, zeros_like
import scipy.cluster.vq as vq
import cv2 as cv
from sklearn import svm, preprocessing
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import RandomizedSearchCV

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
DATASETPATH = '../dataset'
PRE_ALLOCATION_BUFFER = 1000  # for sift
HISTOGRAMS_FILE = 'trainingdata.svm'
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODEBOOK_FILE = 'codebook.file'



def get_categories(datasetpath):
    cat_paths = [files
                 for files in glob(datasetpath + "/*")
                  if isdir(files)]
    cat_paths.sort()
    cats = [basename(cat_path) for cat_path in cat_paths]
    return cats


def get_imgfiles(path):
    all_files = []
    all_files.extend([join(path, basename(fname))
                    for fname in glob(path + "/*")
                    if splitext(fname)[-1].lower() in EXTENSIONS])
    return all_files


def extractSift(input_files):
    print("extracting Sift features")
    all_features_dict = {}
    for i, fname in enumerate(input_files):
        features_fname = fname + '.sift'
        if exists(features_fname) == False:
            print("calculating sift features for", fname)
            sift.process_image(fname, features_fname)
        print("gathering sift features for", fname)
        img1 = cv.imread(fname, 0)  # queryImage
        # Initiate SIFT detector
        # orb = cv.ORB_create(150)
        sift = cv.xfeatures2d.SIFT_create(150)
        # sift = cv.SIFT_create(150)
        # t = cv.xfeatures2d.SIFT_create(200)
        # Compute keypoints
        # locs, descriptors = orb.detectAndCompute(img1, None)
        locs,descriptors = sift.detectAndCompute(img1,None)
        # locs, descriptors = sift.read_features_from_file(features_fname)
        print(descriptors.shape)
        all_features_dict[fname] = descriptors
    return all_features_dict

number_of_descriptors = 128
def dict2numpy(dict):
    nkeys = len(dict)
    array = zeros((nkeys * PRE_ALLOCATION_BUFFER, number_of_descriptors))
    pivot = 0
    for key in dict.keys():
        value = dict[key]
        nelements = value.shape[0]
        while pivot + nelements > array.shape[0]:
            padding = zeros_like(array)
            array = vstack((array, padding))
        array[pivot:pivot + nelements] = value
        pivot += nelements
    array = resize(array, (pivot, number_of_descriptors))
    return array


def computeHistograms(codebook, descriptors):
    code, dist = vq.vq(descriptors, codebook)
    histogram_of_words, bin_edges = histogram(code,
                                              bins=range(codebook.shape[0] + 1),
                                              normed=True)
    return histogram_of_words


def writeHistogramsToFile(nwords, labels, fnames, all_word_histgrams, features_fname):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print('nclusters have been reduced to ' + str(nwords))
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    fmt = '%i '
    for i in range(nwords):
        fmt = fmt + str(i) + ':%f '
    savetxt(features_fname, data_rows, fmt)


def stackHistogramData(nwords, labels, fnames, all_word_histgrams):
    data_rows = zeros(nwords + 1)  # +1 for the category label
    for fname in fnames:
        histogram = all_word_histgrams[fname]
        if (histogram.shape[0] != nwords):  # scipy deletes empty clusters
            nwords = histogram.shape[0]
            data_rows = zeros(nwords + 1)
            print('nclusters have been reduced to ' + str(nwords))
        data_row = hstack((labels[fname], histogram))
        data_rows = vstack((data_rows, data_row))
    data_rows = data_rows[1:]
    return data_rows


path = "/mnt/hgfs/PycharmProjects/objectdetectioncamera/corpus/train/"
if __name__ == '__main__':
    print("---------------------")
    print("## loading the images and extracting the sift features")
    datasetpath = path
    cats = get_categories(datasetpath)
    ncats = len(cats)
    print("searching for folders at " + datasetpath)
    if ncats < 1:
        raise ValueError('Only ' + str(ncats) + ' categories found. Wrong path?')
    print("found following folders / categories:")
    print(cats)
    print("---------------------")
    all_files = []
    all_files_labels = {}
    all_features = {}
    cat_label = {}
    for cat, label in zip(cats, range(ncats)):
        cat_path = join(datasetpath, cat)
        cat_files = get_imgfiles(cat_path)
        cat_features = extractSift(cat_files)
        all_files = all_files + cat_files
        all_features.update(cat_features)
        cat_label[cat] = label
        for i in cat_files:
            all_files_labels[i] = label

    print("---------------------")
    print("## computing the visual words via k-means")
    all_features_array = dict2numpy(all_features)
    nfeatures = all_features_array.shape[0]
    nclusters = int(sqrt(nfeatures))
    codebook, distortion = vq.kmeans(all_features_array,
                                             nclusters,
                                             thresh=K_THRESH)

    # with open(datasetpath + CODEBOOK_FILE, 'wb') as f:
    #     _pickle.dump(codebook, f)
    with open(datasetpath+CODEBOOK_FILE,'rb') as f:
        codebook = _pickle.load(f,encoding="latin1")
    print("---------------------")
    print("## compute the visual words histograms for each image")
    all_word_histgrams = {}
    for imagefname in all_features:
        word_histgram = computeHistograms(codebook, all_features[imagefname])
        all_word_histgrams[imagefname] = word_histgram

    print("---------------------")
    print("## write the histograms to file to pass it to the svm")
    # writeHistogramsToFile(nclusters,
    #                       all_files_labels,
    #                       all_files,
    #                       all_word_histgrams,
    #                       datasetpath + HISTOGRAMS_FILE)
    X = stackHistogramData(nclusters,
                           all_files_labels,
                           all_files,
                           all_word_histgrams)
    pickle.dump(X,open("xfeatures_array.pickle","wb"))
    all = pickle.load(open("xfeatures_array.pickle","rb"),encoding="latin1")
    X = all[:,1:]
    y = all[:,0]
    print("---------------------")
    print("## train svm")
    # randomized_search_params = {'C': scipy.stats.expon(), 'gamma': scipy.stats.expon(scale=.1),
    #  'kernel': ['rbf'], 'class_weight': ['balanced', None]}
    svm_model = svm.SVC()
    svm_model = GaussianProcessClassifier()
    # svm_model
    X_scaled = preprocessing.scale(X)
    svm_model.fit(X_scaled,y)

    # r = RandomizedSearchCV(svm_model,randomized_search_params,n_jobs=8,n_iter=10)
    # r.fit(X_scaled, y)
    model_file = open("scikit_svm.model","wb")
    pickle.dump(svm_model,model_file)
    # c, g, rate, model_file = libsvm.grid(datasetpath + HISTOGRAMS_FILE,
    #                                      png_filename='grid_res_img_file.png')

    print("--------------------")
    print("## outputting results")
    print("model file: " + datasetpath + str(model_file))
    print("codebook file: " + datasetpath + CODEBOOK_FILE)
    print("category      ==>  label")
    for cat in cat_label:
        print('{0:13} ==> {1:6d}'.format(cat, cat_label[cat]))
