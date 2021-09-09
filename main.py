#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Muhammad Bilal Shaikh"
__copyright__ = "Copyright 2021, AR-ResNetX"
__license__ = "GPL"
__version__ = "1.0.1"
__email__ = "mbs.techy@gmail.com"
__status__ = "Production"


import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import torch.optim as optim


# import torchvision module to handle image manipulation

import torchvision
from torchvision import transforms
import torchvision.transforms as transforms
from torch.autograd import Variable
import configparser

# calculate train time, writing train data to files etc.

import time
import numpy as np

from sklearn.svm import SVC

def extract_features(model, dl):
    lbls = []
    model.eval()
    device = 'cuda:0'
    model.cuda(device)
    with torch.no_grad():
        features = None
        for batch in tqdm(dl, disable=True):

            images = batch[0]
            labels = batch[1]
            images = images.to(device)

            # labels = labels.to(device)

            output = model(images)
            lbls.append(labels)

            # print(labels)

            if features is not None:
                features = torch.cat((features, output), 0)
            else:

                features = output

    return (features.cpu().numpy(), lbls)


def flatten_list(t):
    flat_list = [item for sublist in t for item in sublist]
    flat_list = np.array(flat_list)
    return flat_list


def train(TRAIN_DATA_PATH, TEST_DATA_PATH, HPARAM):
    print(torch.__version__)
    print(torchvision.__version__)

    # set device

    if torch.cuda.is_available():
        device = torch.device(('cuda:0'
                               if torch.cuda.is_available() else 'cpu'))
        print (torch.cuda.get_device_properties(device),
               torch.cuda.set_device(device),
               torch.cuda.current_device())

        transform = transforms.Compose([transforms.Resize(299),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(), transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])  # preffered size for network

    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH,
            transform=transform)
    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH,
            transform=transform)

    train_data_loader = torch.utils.data.DataLoader(train_data,
            batch_size=int(HPARAM['batch_size']), shuffle=True,
            num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(test_data,
            batch_size=int(HPARAM['batch_size']), shuffle=True,
            num_workers=4)

    # prepare model

    model_name = 'inceptionresnetv2'  # could be fbresnet152 or inceptionresnetv2
    model = pretrainedmodels.__dict__[model_name](num_classes=1000,
            pretrained='imagenet')

    model.last_linear = nn.Identity()  # freeze the model

    # num_ftrs = model.last_linear.in_features
    # model.last_linear = nn.Linear(num_ftrs, 50)

    for p in model.parameters():
        p.requires_grad = False

    # num_ftrs = model.last_linear.in_features

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model.fc = nn.Linear(num_ftrs, 50)

    PATH = 'models/IRv2.pt'

    torch.save(model, PATH)

    # model = torch.load(PATH)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    def train(model, loader, epochs=1):
        model.to(device)
        model.train()
        print('Training...')
        for epoch in range(epochs):
            start = time.time()
            model.train()
            running_loss = 0

            for (i, batch) in enumerate(loader, int(HPARAM['epoch'])):
                images = batch[0]
                labels = batch[1]
                images = images.to(device)
                labels = labels.to(device)

                preds = model(images)
                loss = F.cross_entropy(preds, labels)  # Adam, SGD, RSPROP

                optimizer.zero_grad()
                loss = Variable(loss, requires_grad=True)
                loss.backward()
                optimizer.step()

                running_loss += loss.data

                if i % 10 == 9:
                    end = time.time()

                    # print ('[epoch %d,imgs %5d] time: %0.3f s'%(epoch+1,(i+1)*4,(end-start)))
                    # print("[epoch %d,imgs %5d] loss: %.7f  time: %0.3f s" % (epoch + 1, (i + 1) * 4, running_loss / 100, (end - start))                    )
                    # tb.add_scalar('Loss', loss, epoch+1)

                    start = time.time()
                    running_loss = 0

    train(model, train_data_loader)

    print('Extracting Features...')
    (train_feat, train_lbls) = extract_features(model,
            train_data_loader)
    (test_feat, test_lbls) = extract_features(model, test_data_loader)
    return (train_feat, train_lbls, test_feat, test_lbls)


    # randomforest, logisticregression, SVM , KNN, LD,

def get_vis_features():

    trainpath = \
        '/home/muhammadbsheikh/workspace/projects/mmaction/mmaction2/train_ucf50_feature.pkl'
    testpath = \
        '/home/muhammadbsheikh/workspace/projects/mmaction/mmaction2/test_ucf50_feature.pkl'

    trainfile = open(trainpath, 'rb')
    testfile = open(testpath, 'rb')

    return (np.array(pickle.load(trainfile)),
            np.array(pickle.load(testfile)))


def svm(
    X,
    Y,
    x_lbls,
    y_lbls,
    ):
    print ('Train-Without FFT')
    svm = SVC(kernel='linear').fit(X, x_lbls)
    preds = svm.predict(Y)
    print ('SVM Accuracy:', metrics.accuracy_score(y_lbls, preds))

    knn_clf = KNeighborsClassifier(n_neighbors=3).fit(X, x_lbls)
    knn_preds = knn_clf.predict(Y)
    print ('KNN Accuracy:', metrics.accuracy_score(y_lbls, knn_preds))

    # Random Forest

    rf_clf = RandomForestClassifier(n_estimators=100).fit(X, x_lbls)
    rf_preds = rf_clf.predict(Y)
    print ('RF Accuracy:', metrics.accuracy_score(y_lbls, rf_preds))

def config():
    settings = configparser.ConfigParser()
    settings._interpolation = configparser.ExtendedInterpolation()
    settings.read(os.getcwd() + '/config.ini')

    # print(settings.sections())

    trainpath = settings.get('data', 'trainpath')
    testpath = settings.get('data', 'testpath')
    os.environ['CUDA_VISIBLE_DEVICES'] = settings.get('sys', 'gpu')
    torch.set_printoptions(linewidth=120)
    torch.set_grad_enabled(True)  # On by default, leave it here for clarity
    os.chdir(settings.get('sys', 'work_dir'))

    HPARAM = {'epoch': settings.get('hparam', 'epoch'),
              'batch_size': settings.get('hparam', 'batch_size'),
              'l_r': settings.get('hparam', 'l_r')}
    return (trainpath, testpath, HPARAM)


def main():

    # Get audio features

    (trainpath, testpath, HPARAM) = config()
    (train_feat, train_lbls, test_feat, test_lbls) = train(trainpath,
            testpath, HPARAM)
    train_lbls = flatten_list(train_lbls)
    test_lbls = flatten_list(test_lbls)  # flatting the lbls

    # Get visual features

    (train_vid_feat, test_vid_feat) = get_vis_features()

    # show shapes

    print (train_feat.shape, test_feat.shape)
    print (train_vid_feat.shape, test_vid_feat.shape)

    # concatenation

    cat_feat_train = np.concatenate((train_feat, train_vid_feat), axis=1)
    cat_feat_test = np.concatenate((test_feat, test_vid_feat), axis=1)
    print (cat_feat_train.shape)
    print (cat_feat_test.shape)

    # get scores

    svm(cat_feat_train, cat_feat_test, train_lbls, test_lbls)


if __name__ == '__main__':
    main()
