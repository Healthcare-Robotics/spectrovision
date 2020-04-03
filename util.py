import os, sys, csv, h5py
import numpy as np
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

def load_data(parent_directory, data_type, image_preprocess, image_shape, five_mats, crop=False, scio_extended=False, test_set=False):
    materials = ['metal', 'plastic', 'wood', 'paper', 'fabric', 'glass', 'ceramic', 'foam']
    measurements_per_object = 100
    materials_objects = []
    materials_objects.extend([['metal', obj] for obj in ['alumfoil', 'bowl', 'canlarge', 'canrockstar', 'cupwithhandle', 'cyltall', 'muggt', 'panbot', 'pot', 'potlarge', 'spraycan', 'traytall', 'waterbottle']])
    materials_objects.extend([['plastic', obj] for obj in ['bag', 'blackcontainer', 'bottlesip', 'clorox', 'coffeemate', 'cupblack', 'cupblue', 'cups', 'folder', 'plate', 'poncho', 'redcokebottle', 'smartwater']])
    materials_objects.extend([['wood', obj] for obj in ['blocksides', 'bowlcylash', 'bowlwhiteshell', 'boxflagus', 'crate', 'cutboard', 'plank', 'platter', 'pothepta', 'pyramid', 'smallblocks', 'soapblock', 'tray']])
    materials_objects.extend([['paper', obj] for obj in ['bagamazon', 'bagstarbucks', 'cups', 'eggcarton', 'folder', 'napkins', 'newspaper', 'notebookgraph', 'packaging', 'printer', 'saltcontainer', 'tissuebox', 'towelbox']])
    materials_objects.extend([['fabric', obj] for obj in ['beanie', 'cardiganred', 'clothblue', 'collarcheckered', 'collargray', 'jeans', 'shirtgray', 'shirtgt', 'shirtwhite', 'socks', 'sweaterknit', 'towelwhite', 'waistcoat']])
    materials_objects.extend([['glass', obj] for obj in ['bowl', 'cupcurved', 'cuptall', 'cuptallpatterned', 'cuptumbler', 'halfliter', 'jartall', 'mason', 'measuringcup', 'pitcher', 'plate', 'tray', 'waterbottle']])
    materials_objects.extend([['ceramic', obj] for obj in ['bowl', 'bowlgreen', 'bowllarge', 'bowlscratched', 'mugblue', 'mugtan', 'mugwhite', 'plate', 'platecorelle', 'platepatterned', 'potorange', 'trayblack', 'traytall']])
    materials_objects.extend([['foam', obj] for obj in ['blacktwo', 'blockstyro', 'cup', 'drygreen', 'piecewhite', 'plate', 'plushies', 'sandal', 'soundpointy', 'spongepatterned', 'styrohead', 'tubes', 'yogamat']])

    if test_set:
        materials_objects = []
        materials_objects.extend([['metal', obj] for obj in ['bowllarge', 'cancampbell', 'mugbubba', 'trayshort', 'traywithhandles']])
        materials_objects.extend([['plastic', obj] for obj in ['bagtarget', 'bottlechew', 'bottlegt', 'coffeered', 'vaseline']])
        materials_objects.extend([['wood', obj] for obj in ['bowllarge', 'boxsoap', 'ladle', 'pot', 'servingbowl']])
        materials_objects.extend([['paper', obj] for obj in ['book', 'googlecardboard', 'napkinsbrown', 'plate', 'starbuckscup']])
        materials_objects.extend([['fabric', obj] for obj in ['hospitalgown', 'pantsblue', 'pillowcase', 'shirtirim', 'shortscargo']])
        materials_objects.extend([['glass', obj] for obj in ['bottlebeer', 'jarfrosted', 'jarrect', 'platepatterned', 'stagioni']])
        materials_objects.extend([['ceramic', obj] for obj in ['muggt', 'potgray', 'traycurved', 'trayshort', 'trayterracota']])
        materials_objects.extend([['foam', obj] for obj in ['evasheet', 'sheet', 'sheetyellow', 'sponge', 'yogablock']])

    data_filename = os.path.join(parent_directory, 'traintest_spectral_data.h5' if data_type == 'spectral' else 'traintest_spectral_image_%s_data.h5' % image_preprocess)
    if test_set:
        data_filename = os.path.join(parent_directory, 'testset_spectral_data.h5' if data_type == 'spectral' else 'testset_spectral_image_%s_data.h5' % image_preprocess)

    if os.path.isfile(data_filename):
        hf = h5py.File(data_filename, 'r')
        X_spectral = np.array(hf.get('X_scio'))
        X_image = np.array(hf.get('X_image'))
        y = np.array(hf.get('y'))
        objs = np.array([n.decode('ascii') for n in hf.get('objs')])
        wavelengths = np.array(hf.get('wavelengths'))
        hf.close()
        # If only five_mats, then delete the glass, ceramic, and foam objects
        if five_mats:
            indices = [i for i, key in enumerate(objs) if ('glass' not in key and 'ceramic' not in key and 'foam' not in key)]
            X_spectral = X_spectral[indices]
            if len(X_image) > 0:
                X_image = X_image[indices]
            y = y[indices]
            objs = objs[indices]
        X_spectral = np.array(X_spectral, dtype=np.float32)
        if 'image' in data_type:
            X_image = np.array(X_image, dtype=np.float32)
        else:
            X_image = np.array(X_spectral, dtype=np.float32)
        return X_spectral, X_image, y, objs, wavelengths

    # Needed for downloading Keras resnet and densenet pretrained models
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    import keras, cv2
    from keras.layers import Flatten, GlobalAveragePooling2D, AveragePooling2D
    from keras_applications import resnet50, densenet, resnext, inception_resnet_v2, nasnet, vgg19, xception, resnet_v2, resnet

    # Load spectral data
    filename = os.path.join(parent_directory, 'spectraldata.csv')
    spectral_data = dict()
    wavelength_count = 331
    wavelengths = None
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 10 or i == 11:
                continue
            if i == 10:
                # Header row
                wavelengths = [float(r.strip().split('_')[-1].split()[0]) + 740.0 for r in row[10:wavelength_count+10]]
                if scio_extended:
                    wavelengths.extend([float(r.strip().split('_')[-1].split()[0]) + 740.0 for r in row[672:wavelength_count+672]])
                continue
            obj = row[3].strip()
            material = row[4].strip()
            key = material + '_' + obj
            if key not in spectral_data:
                spectral_data[key] = []
            spectral_data[key].append([float(v) for v in row[10:wavelength_count+10]])
            if scio_extended:
                spectral_data[key][-1].extend([float(v) for v in row[672:wavelength_count+672]])

    # Load spectral data and images
    X_spectral = []
    X_image = []
    y = []
    objs = []

    if 'image' in data_type and 'none' not in image_preprocess:
        if 'resnet50' in image_preprocess:
            preprocess_model = resnet.ResNet50(weights='imagenet', include_top=False, input_shape=(image_shape[0], image_shape[1], 3), pooling='avg', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        elif 'vgg19' in image_preprocess:
            preprocess_model = vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(image_shape[0], image_shape[1], 3), pooling='avg', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        elif 'densenet201' in image_preprocess:
            preprocess_model = densenet.DenseNet201(weights='imagenet', include_top=False, input_shape=(image_shape[0], image_shape[1], 3), pooling='avg', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        elif 'resnext101' in image_preprocess:
            preprocess_model = resnext.ResNeXt101(weights='imagenet', include_top=False, input_shape=(image_shape[0], image_shape[1], 3), pooling='avg', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
        elif 'resnet152' in image_preprocess:
            preprocess_model = resnet.ResNet152(weights='imagenet', include_top=False, input_shape=(image_shape[0], image_shape[1], 3), pooling='avg', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

    for material, obj in materials_objects:
        key = material + '_' + obj
        X_image_raw = []
        for iteration in range(measurements_per_object):
            if 'image' in data_type:
                filename = os.path.join(parent_directory, 'data_sets_ABCD', material, obj, 'spectrovis_%s_%s_%04d.png' % (material, obj, iteration))
                if not os.path.isfile(filename):
                    print('Image file not found!', filename)
                    exit()
                image = cv2.imread(filename, cv2.IMREAD_COLOR)
                if crop:
                    h, w, _ = np.shape(image)
                    X_image_raw.append(np.copy(image[h//2-image_shape[0]//2:h//2+image_shape[0]//2, w//2-image_shape[1]//2:w//2+image_shape[1]//2]))
                else:
                    X_image_raw.append(cv2.resize(image, (image_shape[1], image_shape[0])))
                # cv2.imshow('Image', X_image[-1])
                # cv2.waitKey(0)
                # if len(X_image) % 50 == 0:
                #     print(len(X_image))
                #     sys.stdout.flush()
            X_spectral.append(spectral_data[key][iteration])
            y.append(materials.index(material))
            objs.append(key)
            if len(X_image_raw) >= 100:
                if 'image' in data_type and 'none' not in image_preprocess:
                    print('X_image_raw:', len(X_image_raw), np.shape(X_image_raw))
                    if 'resnet50' in image_preprocess:
                        X_image_raw = resnet.preprocess_input(np.array(X_image_raw), backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
                    elif 'vgg19' in image_preprocess:
                        X_image_raw = vgg19.preprocess_input(np.array(X_image_raw), backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
                    elif 'densenet201' in image_preprocess:
                        X_image_raw = densenet.preprocess_input(np.array(X_image_raw), backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
                    elif 'resnext101' in image_preprocess:
                        X_image_raw = resnext.preprocess_input(np.array(X_image_raw), backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
                    elif 'resnet152' in image_preprocess:
                        X_image_raw = resnet.preprocess_input(np.array(X_image_raw), backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
                    X_image.extend(preprocess_model.predict(X_image_raw))
                    print('X_image:', len(X_image), np.shape(X_image))
                    sys.stdout.flush()
                    X_image_raw = []

    X_spectral = np.array(X_spectral)
    X_image = np.array(X_image)
    y = np.array(y)
    objs = np.array(objs)
    wavelengths = np.array(wavelengths)
    print('Data loaded. Preprocessing images')
    print(np.shape(X_image))
    sys.stdout.flush()

    hf = h5py.File(data_filename, 'w')
    hf.create_dataset('X_scio', data=X_spectral)
    hf.create_dataset('X_image', data=X_image)
    hf.create_dataset('y', data=y)
    hf.create_dataset('objs', data=[n.encode('ascii') for n in objs])
    hf.create_dataset('wavelengths', data=wavelengths)
    hf.close()

    # If only five_mats, then delete the glass, ceramic, and foam objects
    if five_mats:
        indices = [i for i, key in enumerate(objs) if ('glass' not in key and 'ceramic' not in key and 'foam' not in key)]
        X_spectral = X_spectral[indices]
        if len(X_image) > 0:
            X_image = X_image[indices]
        y = y[indices]
        objs = objs[indices]

    X_spectral = np.array(X_spectral, dtype=np.float32)
    if 'image' in data_type:
        X_image = np.array(X_image, dtype=np.float32)
    else:
        X_image = np.array(X_spectral, dtype=np.float32)
    return X_spectral, X_image, y, objs, wavelengths

