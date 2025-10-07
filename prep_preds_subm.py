from datetime import datetime
import numpy as np
import os


# tags are from the example submission files' first column
tags_preds = np.expand_dims(np.loadtxt('./preds/tags.tsv', delimiter='\t', dtype=str), axis=1)

list_of_tfactors = np.array(['ZNF362', 'GABPA', 'NFKB1', 'PRDM5', 'NACC2', 'TIGD3', 'RORB', 'LEF1'])
header_list = ["tags"]
for tfactor in list_of_tfactors[[0,1,2,3,4]]:
    preds_probs = np.expand_dims(np.loadtxt('./preds/'+tfactor+'/'+os.listdir('./preds/'+tfactor)[-1]), axis=1)
    tags_preds = np.concatenate((tags_preds, preds_probs), axis=1)
    header_list.append(tfactor)

np.savetxt(datetime.now().strftime("%Y_%m_%d_%H_%M")+'_CHS_sub.tsv',
           tags_preds, delimiter='\t', fmt='%s', header="\t".join(header_list), comments="")

os.system("wsl.exe gzip "+datetime.now().strftime("%Y_%m_%d_%H_%M")+"_CHS_sub.tsv")
