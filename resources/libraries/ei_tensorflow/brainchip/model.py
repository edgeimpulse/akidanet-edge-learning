import os, shutil, time
import tensorflow as tf
import numpy as np

def convert_akida_model(dir_path, model, model_path, input_shape):
    from cnn2snn import convert
    import akida

    print('Converting to Akida model...')
    print('')
    # https://doc.brainchipinc.com/api_reference/cnn2snn_apis.html#convert
    # The input_scaling param works like this:
    # input_akida = input_scaling[0] * input_keras + input_scaling[1]
    # It needs to be matched by a similar conversion when we perform inference.
    input_is_image = False
    input_scaling = None
    if len(input_shape) == 3:
        input_is_image = True
        # Don't set input scaling explicitly if there is a rescaling layer present
        # TODO: Make an explicit function for doing these layer checks
        rescaling = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Rescaling)]
        if len(rescaling) == 0:
            input_scaling = (255, 0)

    model_akida = convert(model, input_is_image=input_is_image, input_scaling=input_scaling)

    model_akida.map(akida.AKD1000())

    model_akida.summary()
    print('Converting to Akida model OK')
    print('')

    print('Saving Akida model...')
    model_akida.save(os.path.join(dir_path, model_path))
    print('Saving Akida model OK...')

def save_akida_model(akida_model, path):
    print('Saving Akida model...', flush=True)
    to_save = tf.keras.models.clone_model(akida_model)
    to_save.save(path)
    print('Saving Akida model OK', flush=True)

def load_akida_model(path):
    import akida
    return akida.Model(path)

def make_predictions(mode, model_path, validation_dataset,
                    Y_test, train_dataset, Y_train, test_dataset, Y_real_test):
    prediction_train = None
    prediction_test = None

    if mode == 'classification':
        prediction = predict(model_path, validation_dataset, len(Y_test))
        if (train_dataset is not None) and (Y_train is not None):
            prediction_train = predict(model_path, train_dataset, len(Y_train))
        if (test_dataset is not None) and (Y_real_test is not None):
            prediction_test = predict(model_path, test_dataset, len(Y_real_test))
    elif mode == 'segmentation':
        prediction = predict_segmentation(model_path, validation_dataset, len(Y_test))
    else:
        raise Exception('Unsupported mode for profiling: ' + mode)

    return prediction, prediction_train, prediction_test

def predict_segmentation(model_path, validation_dataset, dataset_length):
    """Runs an Akida model across a set of inputs"""
    model = load_akida_model(model_path)

    last_log = time.time()

    pred_y = []
    for item, _ in validation_dataset.take(-1):
        item = (item * 255)
        item = np.expand_dims(item, axis=0)
        output = model.predict(item.astype('uint8'))
        output = np.squeeze(output)
        pred_y.append(output)
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)

def predict(model_path, validation_dataset, dataset_length):
    """Runs an Akida model across a set of inputs"""
    model = load_akida_model(model_path)

    last_log = time.time()

    pred_y = []
    for item, label in validation_dataset.take(-1).as_numpy_iterator():
        item = (item * 255).astype('uint8')
        item = np.expand_dims(item, axis=0)
        output = model.predict(item)
        output = np.squeeze(output)
        pred_y.append(output)
        current_time = time.time()
        if last_log + 10 < current_time:
            print('Profiling {0}% done'.format(int(100 / dataset_length * (len(pred_y) - 1))), flush=True)
            last_log = current_time

    return np.array(pred_y)

def get_model_sparsity(model_file, mode, dataset):
    """Returns a sparsity of the model, using dataset as an input.

    Returned sparsity is in percents.
    Please refer to:
    https://doc.brainchipinc.com/api_reference/akida_apis.html#sparsity
    """
    import akida
    model = load_akida_model(model_file)

    input = []
    if mode == 'classification':
        for item, _ in dataset.take(-1).as_numpy_iterator():
            item = (item * 255).astype('uint8')
            input.append(item)
    elif mode == 'segmentation':
        for item, _ in dataset.take(-1):
            item = (item * 255)
            input.append(item)

    try:
        raw_sparsity = akida.evaluate_sparsity(model, np.array(input, np.uint8))
    except Exception as err:
        print("EI_LOG_LEVEL=error ERROR: Can't calculate model sparsity: " + str(err))
        return 0

    # calculate average sparsity for model
    num = 0
    sparsity = 0
    for s in raw_sparsity.values():
        if s is not None:
            sparsity += s
            num += 1
    sparsity /= num
    # return sparsity in %
    sparsity *= 100

    return sparsity

def get_hardware_utilization(model_file):
    """Returns utilization of AKD1000 NSoC.
    
    Returned value is a tuple of: program_size, nps, nodes
    Program size is exact size of the model (not the FBZ file) that need to be stored
    NPs is a number of Neural Processors
    Nodes is a number of used nodes (each node consist of 4 NPs)
    """
    import akida

    try:
        model = akida.Model(model_file)
    except Exception as err:
        print("EI_LOG_LEVEL=error ERROR: Can't map model to AKD1000 NSoC! Can't calculate program size, number of NPs and nodes!")
        print("EI_LOG_LEVEL=error ERROR: " + str(err))
        # report -1 (Flash size AKA program size = N/A), and 0 nodes and NPs
        return -1, 0, 0

    try:
        model.map(akida.AKD1000(), hw_only=True)
    except Exception as err:
        print("EI_LOG_LEVEL=warn WARNING: Requested model can't be fully mapped to hardware. Reason:")
        print("EI_LOG_LEVEL=warn WARNING: " + str(err))
        print("EI_LOG_LEVEL=warn WARNING: Reported program size, number of NPs and nodes may not be accurate!")

    # after mapping model onto the hardware (in our case the virtual AKD1000 NSoC)
    # we can iterate over all sequences to collect their sizes and number of used NPs.
    # Some models couldn't be mapped fully on hardware. In such cases, there will be a sewuences
    # (one or a few model layers) that will be processed on the host CPU, so we can't count them.
    total_nps = 0
    program_size = 0
    # iterate through mapped sequences
    for i, seq in enumerate(model.sequences, start=1):
        # if the sequence is not mapped to hardware, then skip it
        if seq.backend != akida.BackendType.Hardware:
            continue
        # get program size of the current sequence
        program_size += len(seq.program)
        # iterate through passes in sequence
        # see pass info: https://doc.brainchipinc.com/api_reference/akida_apis.html?highlight=passes#akida.Pass
        for j, seq_pass in enumerate(seq.passes, start=1):
            total_nps_pass = 0
            for n, layer in enumerate(seq_pass.layers, start=1):
                try:
                    nps = [conf.ident for conf in layer.mapping.nps]
                except:
                    continue
                if i==1 and j==1 and n==1:
                    # first layer NPs is in the HRC and use default NPs
                    continue
                total_nps_pass += len(nps)
            total_nps += total_nps_pass

    # one node consist of 4 NPs, so if we are using 9 NPs (2,25 node) we need to report 3 nodes
    # Node is a unit used for licensing BrainChip's IP, so it's important to see how many nodes
    # you need to license as a customer
    nodes = int(total_nps / 4)
    if total_nps % 4:
        nodes += 1

    return (program_size, total_nps, nodes)
