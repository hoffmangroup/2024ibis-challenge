import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Time and exp. data (e.g., TFs)
start_time = datetime.datetime.now()
print(start_time)

exp_desc = "zeng_cnn_arch"

# run util. params
test_converted = 0
averages_arr = []

data_file = "CHS_ZNF395_THC_0294_Rep-DIANA_0293_peaks_ohe.npz"[:-4]  # change input source here; todo cli arguments
data_folder = "./data"  # + "/"
newpath = r"./results/" + exp_desc + "/" + data_file + "/" + start_time.strftime("%Y_%m_%d_%H_%M_%S")

# load data
tf_data = np.load(data_folder + "/" + data_file + ".npz")
x_train, y_train, x_test, y_test = tf_data['x_train'], tf_data['y_train'], tf_data['x_test'], tf_data['y_test']
x_train, x_test = np.expand_dims(x_train, 1), np.expand_dims(x_test, 1)

if not os.path.exists(newpath):
    os.makedirs(newpath)

for se in range(1):  # increase to have more models for measuring std or ensembling
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(256, (1, 24), input_shape=(x_train.shape[1:]),
                      activity_regularizer=tf.keras.regularizers.l1(5e-5),
                      kernel_regularizer=tf.keras.regularizers.l2(0.0005), padding='same',),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(64, (1, 12), padding='same', activity_regularizer=tf.keras.regularizers.l1(5e-5),
                      kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.GlobalMaxPooling2D(),  # O: pool_size=(2, 2)
        tf.keras.layers.Dense(250, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(lr=5e-5)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=5, verbose=1,
                                                     mode='auto', min_delta=0.0001, cooldown=0, min_lr=0, )

    summary_out = model.summary()

    epochs = 100

    history = model.fit(x_train, y_train, epochs=epochs, verbose=2, validation_split=0.1,
                        batch_size=64, callbacks=[early_stopping, reduce_lr])

    hist = pd.DataFrame(history.history)
    print(hist.tail())

    print("Training finished at:", datetime.datetime.now())

    # Evaluation - confusion matrix - test set
    preds_test = model.predict(x_test, verbose=2)
    preds_test = np.argmax(preds_test, axis=1)
    tn, fp, fn, tp = metrics.confusion_matrix(np.argmax(y_test, axis=1), preds_test).ravel()

    eval_score = model.evaluate(x_test, y_test, verbose=2)
    print("Test accuracy: ", eval_score[1])

    val_max = np.max(hist.values[:, 3])
    val_max = format(val_max, '.6f')
    print("Validation maximum at run %d: " % se, val_max)

    d = {'test acc': [format(eval_score[1], '.6f')],
         'val acc': [val_max],
         'train acc': [np.max(hist.values[:, 1])],
         'iter': [se],
         # 'optimized' : [optimized_param],
         # 'optimized2' : [optimized_param2],
         # 'param test': [param_test],
         'epochs': [hist.index[-1]],
         # 'changes': [changes],

         }

    df = pd.DataFrame(data=d, columns=['test acc', 'val acc', 'train acc', 'iter', 'param test', 'epochs', 'changes', ])
    # 'optimized','optimized2'])

    df.to_csv(newpath + '/dk.csv', index=False, mode='a', header=not os.path.exists(newpath + '/dk.csv'))
    averages_arr.append([format(eval_score[1], '.6f'), val_max, np.max(hist.values[:, 1])])

    model.save(start_time.strftime("%Y_%m_%d_%H_%M_%S")+".keras")

    # test eval prep
    # testdata = np.load('../data/CHS_test.npy')
    # preds = model.predict(np.expand_dims(testdata, axis=1), verbose=2)
    # np.savetxt('./preds/'+data_file.split("_")[1]+'/'+start_time.strftime("%Y_%m_%d_%H_%M_%S")+"preds.txt", preds[:,-1], fmt="%0.5f")
    # print("Saved test predictions from "+data_file+".txt")

averages_np = np.array(averages_arr, dtype=float)
print(np.mean(averages_np, axis=0))
# save model structure 01
# struct_network = " ".join([struct_network, str(s2), exp_desc],)
# np.savetxt(newpath + "/model_complexity.txt", [struct_network], fmt='%s')

# Time running
end_time_of_script = datetime.datetime.now()
print("Start 'Time of run': ", start_time.strftime("%Y_%m_%d_%H_%M_%S"))
print("Script finished at: ", end_time_of_script)
print("Running time: ", end_time_of_script - start_time)
