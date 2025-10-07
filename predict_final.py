import tensorflow as tf
import numpy as np


""" start_time_tf_model.tsv should contain a list of .keras models trained for the different TFs, for example:
USF3    2024_08_15_07_36_36.keras
ZBED2   2024_08_15_07_41_21.keras
"""
model_names_date_times_tfs = np.loadtxt("start_time_tf_model.tsv", delimiter="\t", dtype=str)
data_file = "PBM_test.npy"  # replace this string with the desired input format (e.g., fasta, pbm)

# get tags fasta:
# tags_preds = np.loadtxt('../data/IBIS.test_data.Final.v1/' + data_file.split("_")[0] + '_participants.fasta', delimiter='\t', dtype=str)
# tags_preds = np.array([a[1:] for a in tags_preds if ">" in a], ndmin=2).T
# get tags PBM fasta:
tags_preds = np.loadtxt('../data/IBIS.test_data.Final.v1/' + data_file.split("_")[0] + '_participants.fasta', delimiter='\t', dtype=str)

# HTS
testdata1 = np.load('../data/' + data_file)
testdata1 = testdata1.astype(bool)
testdata = np.full((testdata1.shape[0],100,4), False)
testdata[:,30:70,:] = testdata1
testdata = np.expand_dims(testdata, axis=1)


for model_name in model_names_date_times_tfs[:,1]:

    model = tf.keras.models.load_model(model_name)
    preds = model.predict(testdata, verbose=1)
    tags_preds = np.concatenate([tags_preds, np.round(preds[:,-1,np.newaxis],5)], axis=1)

np.savetxt('./preds/' + data_file.split(".")[0] + "_preds.tsv", tags_preds, fmt='%s', comments="",
           header="\t".join(['tag',*model_names_date_times_tfs[:,0]]), delimiter='\t', )
print("Saved test predictions from " + data_file + ".tsv")





#