def quantize_brainchip(model,
                       train_dataset: tf.data.Dataset,
                       validation_dataset: tf.data.Dataset,
                       best_model_path: str, optimizer: str,
                       fine_tune_loss: str,
                       fine_tune_metrics: 'list[str]',
                       callbacks, stopping_metric='val_accuracy',
                       verbose=2):
    import tensorflow as tf
    import cnn2snn

    print('Performing post-training quantization...')

    akida_model = cnn2snn.quantize(model,
                            weight_quantization=4,
                            activ_quantization=4,
                            input_weight_quantization=8)
    print('Performing post-training quantization OK')
    print('')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=stopping_metric,
                                                      mode='max',
                                                      verbose=1,
                                                      min_delta=0,
                                                      patience=10,
                                                      restore_best_weights=True)
    callbacks.append(early_stopping)

    print('Running quantization-aware training...')
    akida_model.compile(optimizer=optimizer,
                    loss=fine_tune_loss,
                    metrics=fine_tune_metrics)

    akida_model.fit(train_dataset,
                    epochs=30,
                    verbose=verbose,
                    validation_data=validation_dataset,
                    callbacks=callbacks)

    print('Running quantization-aware training OK')
    print('')

    return akida_model
