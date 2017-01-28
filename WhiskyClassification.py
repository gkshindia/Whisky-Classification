# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:31:10 2017

@author: KANHAIYA
"""
from sklearn.cluster.bicluster import SpectralCoclustering
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


whisky = pd.read_csv("whiskies.csv")

whisky["Region"] = pd.read_csv("regions.csv")
flavours = whisky.iloc[: , 2:14]
corr_flavors = pd.DataFrame.corr(flavours)
plt.figure(figsize = (10,10))
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("corr_flavor.pdf")

corr_whisky = pd.DataFrame.corr(flavours.transpose())
plt.figure(figsize = (10,10))
plt.pcolor(corr_whisky)
plt.colorbar()
plt.savefig("corr_whisky.pdf")

model = SpectralCoclustering(n_clusters = 6, random_state = 0)
model.fit(corr_whisky)
whisky['Group'] = pd.Series(model.row_labels_, index = whisky.index)
whisky = whisky.ix[np.argsort(model.row_labels_)]
whisky = whisky.reset_index(drop = True)
correlations = pd.DataFrame.corr(whisky.iloc[:, 2:14].transpose())
correlations = np.array(correlations)

plt.figure(figsize = (14,7))
plt.subplot(121)
plt.pcolor(corr_whisky)
plt.title("Original")
plt.axis("tight")
plt.subplot(122)
plt.pcolor(correlations)
plt.title("Rearranged")
plt.axis("tight")
plt.colorbar()
plt.savefig("correlations.pdf")