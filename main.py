import os, random, warnings, argparse
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import numpy as np

seed_value = 1000
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
torch.manual_seed(seed_value)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

import sys, pickle, time, gc
from poutyne.framework import Model
import multiprocessing
from functools import partial
from multiprocessing.pool import Pool
import util

from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

def first_deriv(x, wavelengths):
    # First derivative of measurements with respect to wavelength
    x = np.copy(x)
    for i, xx in enumerate(x):
        dx = np.zeros(xx.shape, np.float)
        dx[0:-1] = np.diff(xx)/np.diff(wavelengths)
        dx[-1] = (xx[-1] - xx[-2])/(wavelengths[-1] - wavelengths[-2])
        x[i] = dx
    return x

def prepare_data(x_spectral_train, x_image_train, y_train, x_spectral_test, x_image_test, y_test, wavelengths, deriv=True, data_type='spectral_image', image_preprocess='resnet', objs=None):
    if deriv:
        # Finite difference (Numerical differentiation)
        x_spectral_train = np.concatenate([x_spectral_train, first_deriv(x_spectral_train, wavelengths)], axis=-1)
        x_spectral_test = np.concatenate([x_spectral_test, first_deriv(x_spectral_test, wavelengths)], axis=-1)

    if data_type == 'spectral':
        # Zero mean unit variance
        scaler_spectral = preprocessing.StandardScaler()
        x_spectral_train = scaler_spectral.fit_transform(x_spectral_train)
        x_spectral_test = scaler_spectral.transform(x_spectral_test)
    elif data_type == 'image':
        if 'none' in image_preprocess:
            scaler_image = preprocessing.StandardScaler()
            x_image_train = np.reshape(scaler_image.fit_transform(np.reshape(x_image_train, (len(x_image_train), -1))), np.shape(x_image_train))
            x_image_test = np.reshape(scaler_image.transform(np.reshape(x_image_test, (len(x_image_test), -1))), np.shape(x_image_test))
        else:
            scaler_image = preprocessing.StandardScaler()
            x_image_train = scaler_image.fit_transform(x_image_train)
            x_image_test = scaler_image.transform(x_image_test)
    elif data_type == 'spectral_image':
        scaler_image = preprocessing.StandardScaler()
        x = scaler_image.fit_transform(np.concatenate([x_spectral_train, x_image_train], axis=-1))
        x_spectral_train = x[:, :np.shape(x_spectral_train)[-1]]
        x_image_train = x[:, np.shape(x_spectral_train)[-1]:]
        x = scaler_image.transform(np.concatenate([x_spectral_test, x_image_test], axis=-1))
        x_spectral_test = x[:, :np.shape(x_spectral_test)[-1]]
        x_image_test = x[:, np.shape(x_spectral_test)[-1]:]

    return x_spectral_train, x_image_train, y_train, x_spectral_test, x_image_test, y_test

def nn(input_size, layers, dropout, material_count=None, batchnorm=True):
    modules = []
    for i in range(len(layers)-1):
        modules.append(torch.nn.Linear(input_size if i == 0 else layers[i-1], layers[i]))
        if batchnorm:
            modules.append(torch.nn.BatchNorm1d(layers[i]))
        modules.append(torch.nn.LeakyReLU())
        if dropout > 0:
            modules.append(torch.nn.Dropout(dropout))
    modules.append(torch.nn.Linear(layers[-2], layers[-1]) if len(layers) > 1 else torch.nn.Linear(input_size, layers[-1]))
    if batchnorm:
        modules.append(torch.nn.BatchNorm1d(layers[-1]))
    modules.append(torch.nn.LeakyReLU())
    if material_count is not None:
        modules.append(torch.nn.Linear(layers[-1], material_count))
    return torch.nn.Sequential(*modules)

def learn(o, X_spectral, X_image, y, objs, wavelengths, data_type='spectral_image', image_preprocess='resnet', epochs=50, batch_size=128, material_count=8, layers=[64]*2, dropout=0.0, lr=0.0005, seed=1000, test='looo', verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)

    if 'test' in test:
        train_idx, test_idx = o
        X_spectral_train = X_spectral[train_idx]
        X_image_train = X_image[train_idx]
        y_train = y[train_idx]
        X_spectral_test = X_spectral[test_idx]
        X_image_test = X_image[test_idx]
        y_test = y[test_idx]
        objs_train = objs[train_idx]
    elif 'looo' in test:
        _, obj = o
        # Set up leave-one-object-out training and test sets
        X_spectral_train = X_spectral[objs != obj]
        X_image_train = X_image[objs != obj]
        y_train = y[objs != obj]
        X_spectral_test = X_spectral[objs == obj]
        X_image_test = X_image[objs == obj]
        y_test = y[objs == obj]
        objs_train = objs[objs != obj]

    X_spectral_train, X_image_train, y_train, X_spectral_test, X_image_test, y_test = prepare_data(X_spectral_train, X_image_train, y_train, X_spectral_test, X_image_test, y_test, wavelengths, deriv=True, data_type=data_type, image_preprocess=image_preprocess, objs=objs)

    if data_type == 'spectral':
        X_spectral_train, y_train = shuffle(X_spectral_train, y_train)
    elif data_type == 'image':
        X_image_train, y_train = shuffle(X_image_train, y_train)
    elif data_type == 'spectral_image':
        X_spectral_train, X_image_train, y_train = shuffle(X_spectral_train, X_image_train, y_train)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Image layers
    if data_type == 'spectral_image':
        spectral_layers = [64, 64, 32, 32]
        spectral_dropout = 0.25
        spectral_epochs = 50
        image_layers = [128, 64, 32]
        image_dropout = 0.1
        image_epochs = 50

        class ConcatPretrainedNetwork(torch.nn.Module):
            def __init__(self):
                # global spectral_accuracy, image_accuracy
                super().__init__()
                spectral_net = nn(np.shape(X_spectral_train)[-1], spectral_layers, spectral_dropout, material_count)
                opt = torch.optim.Adam(spectral_net.parameters(), lr=lr)
                spectral_model = Model(spectral_net, opt, 'cross_entropy', batch_metrics=['accuracy'])
                spectral_model.fit(X_spectral_train, y_train, epochs=spectral_epochs, batch_size=batch_size, verbose=False)

                image_net = nn(np.shape(X_image_train)[-1], image_layers, image_dropout, material_count)
                opt = torch.optim.Adam(image_net.parameters(), lr=lr)
                image_model = Model(image_net, opt, 'cross_entropy', batch_metrics=['accuracy'])
                image_model.fit(X_image_train, y_train, epochs=image_epochs, batch_size=batch_size, verbose=False)

                # Disable dropout, remove last layer and freeze network
                self.trained_spectral_model = torch.nn.Sequential(*(list(spectral_net.children())[:-1]))
                for p in self.trained_spectral_model.parameters():
                    p.requires_grad = False
                self.trained_image_model = torch.nn.Sequential(*(list(image_net.children())[:-1]))
                for p in self.trained_image_model.parameters():
                    p.requires_grad = False

                self.concat_net = nn(spectral_layers[-1] + image_layers[-1], layers, dropout, material_count, batchnorm=False)
            def forward(self, x_spectral, x_image):
                y1 = self.trained_spectral_model(x_spectral)
                y2 = self.trained_image_model(x_image)
                concat = torch.cat((y1, y2), -1)
                return self.concat_net(concat)
        net = ConcatPretrainedNetwork()
        X_train = [X_spectral_train, X_image_train]
        X_test = [X_spectral_test, X_image_test]
    elif data_type == 'image':
        net = nn(np.shape(X_image_train)[-1], layers, dropout, material_count)
        X_train = X_image_train
        X_test = X_image_test
    elif data_type == 'spectral':
        net = nn(np.shape(X_spectral_train)[-1], layers, dropout, material_count)
        X_train = X_spectral_train
        X_test = X_spectral_test

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    model = Model(net, opt, 'cross_entropy', batch_metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=False)
    cm = confusion_matrix(y_test, model.predict(X_test).argmax(axis=-1), labels=range(material_count))
    # Return accuracy and confusion matrices
    if 'backprop' in test:
        return {'accuracy': model.evaluate(X_test, y_test)[-1], 'cm': cm, 'obj_cm': np.copy(cm[y_test[0]]), 'model': model, 'net': net, 'X_test': X_test, 'y_test': y_test}
    else:
        if verbose:
            print(obj, model.evaluate(X_test, y_test)[-1])
        return {'accuracy': model.evaluate(X_test, y_test)[-1], 'cm': cm, 'obj_cm': np.copy(cm[y_test[0]])}

# NOTE: Leave-one-object-out cross-validation.

def looo_cv(X_spectral, X_image, y, objs, wavelengths, data_type, image_preprocess, layers, dropout, epochs, batch_size, lr, seeds, material_count, jobs, test='looo', verbose=False):
    objs_set = []
    for o in objs:
        if o not in objs_set:
            objs_set.append(o)
    accuracies = []
    confusion_mat = None
    object_confusion_matrix = []
    for s in seeds:
        pool = Pool(processes=jobs)
        results = pool.imap(partial(learn, X_spectral=X_spectral, X_image=X_image, y=y, objs=objs, wavelengths=wavelengths, data_type=data_type, image_preprocess=image_preprocess, epochs=epochs, batch_size=batch_size, material_count=material_count, layers=layers, dropout=dropout, lr=lr, seed=s, test=test, verbose=verbose), list(enumerate(objs_set)))
        pool.close()
        pool.join()

        seed_accuracy = []
        for result in list(results):
            accuracies.append(result['accuracy'])
            seed_accuracy.append(result['accuracy'])
            if confusion_mat is None:
                confusion_mat = result['cm']
            else:
                confusion_mat += result['cm']
            object_confusion_matrix.append(result['obj_cm'])
        print('Accuracy with seed', s, ':', np.mean(seed_accuracy))
    print('Results for:', data_type, '5 materials' if five_mats else '8 materials', image_preprocess, '- Accuracy:', np.mean(accuracies))
    # print(confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis], '\n')
    sys.stdout.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training and Evaluation for SpectroVision')
    parser.add_argument('-s', '--seed', default=-1, help='Seed used for random number generators. (default -1) for averaging of 10 seeds', type=int)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()

    verbose = args.verbose
    jobs = multiprocessing.cpu_count()
    layers = {'spectral': [64, 64, 32, 32], 'image': [128, 64, 32], 'spectral_image': [32]}
    dropout = {'spectral': 0.25, 'image': 0.1, 'spectral_image': 0.0}
    epochs = {'spectral': 50, 'image': 50, 'spectral_image': 10}
    lr = 0.0005
    batch_size = 128
    seeds = range(8000, 8010) if args.seed == -1 else [args.seed]
    parent_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')

    # Table I, Leave-one-object-out
    print('Table I: leave-one-object-out')
    for five_mats in [True, False]:
        for data_type in ['spectral', 'image', 'spectral_image']:
            X_spectral, X_image, y, objs, wavelengths = util.load_data(parent_directory, data_type, 'densenet201_240_320', (240, 320), five_mats)
            # print(np.shape(X_spectral), np.shape(X_image))
            print('Computing results for:', data_type, '5 materials' if five_mats else '8 materials')
            looo_cv(X_spectral, X_image, y, objs, wavelengths, data_type, 'densenet201_240_320', layers[data_type], dropout[data_type], epochs[data_type], batch_size, lr, seeds, 5 if five_mats else 8, jobs, 'looo', verbose)

    # Table II, Test set
    print('Table II: test set')
    for five_mats in [True, False]:
        for data_type in ['spectral', 'image', 'spectral_image']:
            X_spectral, X_image, y, objs, wavelengths = util.load_data(parent_directory, data_type, 'densenet201_240_320', (240, 320), five_mats)
            X_spectral_test, X_image_test, y_test, objs_test, wavelengths_test = util.load_data(parent_directory, data_type, 'densenet201_240_320', (240, 320), five_mats, test_set=True)
            accuracies = []
            confusion_mat = None
            for seed in seeds:
                results = learn([list(range(len(y))), list(range(len(y), len(y)+len(y_test)))], np.concatenate([X_spectral, X_spectral_test], axis=0), np.concatenate([X_image, X_image_test], axis=0), np.concatenate([y, y_test], axis=0), np.concatenate([objs, objs_test], axis=0), wavelengths, data_type, 'densenet201_240_320', epochs[data_type], batch_size, 5 if five_mats else 8, layers[data_type], dropout[data_type], lr, seed, 'test', verbose)
                accuracies.append(results['accuracy'])
                if confusion_mat is None:
                    confusion_mat = results['cm']
                else:
                    confusion_mat += results['cm']
            print(data_type, '5 materials' if five_mats else '8 materials', '- Accuracy:', np.mean(accuracies))
            # print(confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis], '\n')
            sys.stdout.flush()

    # Table IV, Resizing and cropping images
    print('Table IV: resizing and cropping images')
    for image_shape, crop in [((240, 320), False), ((240, 320), True), ((480, 640), False), ((480, 640), True), ((960, 1280), False)]:
        five_mats = False
        data_type = 'image'
        image_preprocess = 'densenet201_%d_%d%s' % (image_shape[0], image_shape[1], '_crop' if crop else '')
        X_spectral, X_image, y, objs, wavelengths = util.load_data(parent_directory, data_type, image_preprocess, (240, 320), five_mats)
        looo_cv(X_spectral, X_image, y, objs, wavelengths, data_type, image_preprocess, layers[data_type], dropout[data_type], epochs[data_type], batch_size, lr, seeds, 5 if five_mats else 8, jobs, 'looo', verbose)

    # Table V, ImageNet models
    print('Table V: ImageNet models')
    for model in ['vgg19', 'resnet50', 'resnet101', 'resnet152', 'densenet201', 'resnext101', 'efficientnet-b5']:
        five_mats = False
        data_type = 'image'
        image_preprocess = ('%s_240_320' % model) if 'efficient' not in model else ('%s_456_608' % model)
        X_spectral, X_image, y, objs, wavelengths = util.load_data(parent_directory, data_type, image_preprocess, (240, 320), five_mats)
        looo_cv(X_spectral, X_image, y, objs, wavelengths, data_type, image_preprocess, layers[data_type], dropout[data_type], epochs[data_type], batch_size, lr, seeds, 5 if five_mats else 8, jobs, 'looo', verbose)

