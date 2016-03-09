from py_stringmatching import tokenizers, simfunctions
import pandas as pd
import numpy as np
amz = pd.read_csv('amazonProds_cleaned.csv')
wal = pd.read_csv('walmartProds_cleaned.csv')

brand = 'brand'
category = 'category two'
title = 'this is the title'

# b - affine, lev, nw, sw, hamming, jaro, jarow,
# c, t - jaccard, overlap, cosine, ml, tfidf, softtfidf

print 'string sims start '




for temp in amz.brand.tolist():
    if temp is not np.NaN:
        simfunctions.affine(temp, brand)
        simfunctions.levenshtein(temp, brand)
        simfunctions.needleman_wunsch(temp, brand)
        simfunctions.smith_waterman(brand, temp)
        simfunctions.hamming_distance(brand, brand)
        simfunctions.jaro(temp, brand)
        simfunctions.jaro_winkler(temp, brand)
# for c in amz.category.tolist():

print 'string sims end '





