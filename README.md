# 2024ibis-challenge
IBIS Challenge code scripts for Pap Team.
G2A-AAA


### Pre-processing notes
HT-Selex: [flexbar](https://github.com/seqan/flexbar) was used to process .fastq files, for example:

`flexbar --reads data/HTS/NACC2/NACC2_R0_C2_lf5ACGACGCTCTTCCGATCTTG_rf3TGTGTTAGATCGGAAGAGCA.fastq  --post-trim-length 40 --min-read-length 20 --qtrim-threshold 30 --output-reads HTS_NACC2_R0_C2.fasta --fasta-output --number-tags --stdout-log > log_flexbar2.txt`

Calling `bed2ntohe.py $input_data` converts the file to an OHE nucleotide representation and saves it as a numpy array to disk.

Creating the entities of the negative class and forming the validation set are done by:<br>`prepare_training.py $input_file`

### Model training
The `cnn_01.py` contains code for model creation and fitting. 

### Submission scripts
The `prep_preds_subm.py` and `predict_final.py` are used for preparing the submissions.
This includes loading the target test input data, and the corresponding tags. Then getting
the probabilities via model inference.