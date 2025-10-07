import os
import sys

import numpy as np
import pandas as pd
from pyfaidx import Fasta

'''
For converting the experimental input file data to nucleotide one-hot encoded representation.
'''

# prepare options
half_length = 50

# prepare chromosomes / fasta files
chrs = ["chr" + str(a) for a in np.arange(1, 23)]

# Get fasta sequences for each chromosome in the 'chrs' list using pyfaidx
chr_dict = {}
for chr_n in chrs:  # chr Number
    chr_fasta_dict = Fasta('./data/ref_genome/'+chr_n+'.fa', one_based_attributes=False)
    chr_seqs = chr_fasta_dict[list(chr_fasta_dict.keys())[0]]  # get sequences based on dict. keys
    chr_dict[chr_n] = chr_seqs  # add ref. sequences for current chr to a dict. for fast access

def nuc2ohe(seq):
    # mapping = dict(zip("ACGT", range(4)))
    # seq2 = [mapping[i] for i in seq]
    # return np.eye(4)[seq2]
    seq2=list()
    mapping = {"A":[1., 0., 0., 0.], "C": [0., 1., 0., 0.], "G": [0., 0., 1., 0.], "T":[0., 0., 0., 1.]}
    for i in seq:
      seq2.append(mapping[i]  if i in mapping.keys() else [0., 0., 0., 0.])
    return np.array(seq2)

nt_feature_vector = []  # the input matrix
# loop over the sequences and gather the mean for the nucleotide sequences

# @profile
def convert1(path_data, tfactor, peaks):  # abs_summit
    peaks_df = pd.read_csv(os.path.join(path_data, tfactor, peaks), sep='\t')

    for posi in peaks_df.index[:]:  # todo extend into a function

        chr_n, summit_peak = peaks_df.loc[posi, '#CHROM'], int(peaks_df.loc[posi, 'abs_summit'])
        start_pos, end_pos = summit_peak - half_length, summit_peak + half_length
        cur_seq = chr_dict[chr_n][start_pos:end_pos].seq  # current sequence for the dinucleotide loop
        nt_feature_vector.append(nuc2ohe(cur_seq.upper()))

    np.save('./data/CHS_' + tfactor + '_' + peaks.replace(".", "_") +
            '_ohe.npy', np.array(nt_feature_vector, dtype=np.float32))
    print("convert1: ./data/CHS_" + tfactor + "_" + peaks.replace(".", "_") + "_ohe.npy saved!")


def convert2_test_chs_bed(df_bed_path):
    df_bed = pd.read_csv(df_bed_path, sep="\t", names=["chr", "start", "end", "tag"])

    for posi in df_bed.index:

        chr_n = df_bed.loc[posi, 'chr']
        start_pos, end_pos = df_bed.loc[posi, 'start'], df_bed.loc[posi, 'end']
        cur_seq = chr_dict[chr_n][start_pos:end_pos].seq  # current sequence for the dinucleotide loop
        nt_feature_vector.append(nuc2ohe(cur_seq.upper()))

    np.save('./data/GHTS_test.npy', np.array(nt_feature_vector, dtype=np.float16))

def merge_replicates(peaks_path):
    repl_list = os.listdir(peaks_path)
    print("Found replicates: ", repl_list)
    r1 = pd.read_csv(peaks_path + repl_list[0], delimiter="\t")
    for repl in repl_list[1:]:
        r2 = pd.read_csv(peaks_path + repl, delimiter="\t",)
        r1 = pd.concat([r1, r2], ignore_index=True)
    r1.to_csv(peaks_path + "THC_merged.peaks", index=False, sep="\t")

def read_fastq(fastq_path):

    def process(q_lines=None):
        ks = ['name', 'sequence', 'optional', 'quality']
        return {k: v for k, v in zip(ks, q_lines)}

    try:
        fn = fastq_path  # sys.argv[1]
    except IndexError as ie:
        raise SystemError("Error: Specify file name\n")

    out_seqs = []
    n = 4
    with open(fn, 'r') as fh:
        lines = []
        for line in fh:
            lines.append(line.rstrip())
            if len(lines) == n:
                record = process(lines)
                # sys.stderr.write("Record: %s\n" % (str(record)))
                lines = []
                out_seqs.append(record["sequence"])
            print("{} fastq sequences converted!".format(len(out_seqs)), end = '\r')
    out_seqs_ohe = [nuc2ohe(seq) for seq in out_seqs]
    np.save(fastq_path[:8] + fastq_path.split("/")[2]+"_"+fastq_path.split("/")[3]+ '_fastq_one.npy', np.array(out_seqs_ohe, dtype=np.float16))
    print(fastq_path[:8] + fastq_path.split("/")[2]+"_"+fastq_path.split("/")[3]+ '_fastq_one.npy saved!')


def read_fasta(fasta_path):
    fasta_list = [x for x in os.listdir(fasta_path) if ".fasta" in x]
    fasta_list = [x for x in fasta_list if "SMS" in x]
    # fasta_list = [x for x in fasta_list if "G" not in x]

    def process(q_lines=None):
        ks = ['name', 'sequence',]
        return {k: v for k, v in zip(ks, q_lines)}

    out_seqs = []
    n = 2

    for fasta_file in fasta_list:
        with open(fasta_path+fasta_file, 'r') as fh:
            lines = []
            for line in fh:
                lines.append(line.rstrip())
                if len(lines) == n:
                    record = process(lines)
                    # sys.stderr.write("Record: %s\n" % (str(record)))
                    lines = []
                    out_seqs.append(record["sequence"])
                print("{} fasta sequences converted!".format(len(out_seqs)), end = '\r')
    out_seqs_ohe = [nuc2ohe(seq) for seq in out_seqs]
    np.save("./data/" + fasta_list[0].split("R")[0] + 'fasta_one.npy', np.array(out_seqs_ohe, dtype=np.uint8))
    print("./data/" + fasta_list[0].split("R")[0] + 'fasta_one.npy saved!')


# call the function of converting
# set_convert = "fasta"  # merge test_bed

def main():
    set_convert = sys.argv[1]
    if set_convert == "test":
        df_bed_path = "../data/IBIS.test_data.Final.v1/GHTS_participants.bed"
        convert2_test_chs_bed(df_bed_path)

    elif set_convert=="merge":
        # rp_ls = ["THC_0307.Rep-DIANA_0293.peaks", "THC_0307.Rep-MICHELLE_0314.peaks",]  # Replicates' List
        peaks_path = "../data/CHS/LEUTX/"
        merge_replicates(peaks_path)
        convert1(path_data = '../data/CHS', tfactor='LEUTX', peaks ="THC_merged.peaks")
    elif set_convert == "fastq":
        read_fastq(fastq_path='../data/HTS/NACC2/NACC2_R0_C3_lf5ACGACGCTCTTCCGATCTTG_rf3TGTGTTAGATCGGAAGAGCA.fastq')
    elif set_convert == "fasta":
        read_fasta(fasta_path='../data/IBIS.test_data.Final.v1/')  # ../data/HTS/NACC2/
    else:
        convert1(path_data = '../data/CHS', tfactor='ZNF395', peaks ='THC_0294.Rep-DIANA_0293.peaks')  # convert1('../data/CHS/ZNF407/THC_0668.peaks')

main()

