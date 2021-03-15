import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Data source: https://www.kaggle.com/fedesoriano/cern-electron-collision-data
#
# Description
# In this analysis, we want to briefly explore the characteristics of the data set and glean inferences about
# the relationships between variables. The variable abbreviations are as follows:
# Run: The run number of the event.
# Event: The event number.
# E1, E2: The total energy of the electron (GeV) for electrons 1 and 2.
# px1, py1, pz1, px2, py2, pz2: The components of the momentum of the electron 1 and 2 (GeV).
# pt1, pt2: The transverse momentum of the electron 1 and 2 (GeV).
# eta1, eta2: The pseudorapidity of the electron 1 and 2.
# phi1, phi2: The phi angle of the electron 1 and 2 (rad).
# Q1, Q2: The charge of the electron 1 and 2.
# M: The invariant mass of two electrons (GeV).

df = pd.read_csv('dielectron.csv')

corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()

# We can see very strong positive correlation between the target M and the variables E1, E2, pt1, and pt2.
# There is also a strong correlation between the y-coordinate and the phi value for each electron, the z-coordinate
# and eta values of the individual electrons, and, within the pair of electrons, the x-coordinate of one is strongly
# negatively correlated with the x-coordinate of the other. The same is true for the y-coordinates of both electrons
# as well.

sns.histplot(df, x='M', bins=40)
plt.title('Histogram of target variable M')
plt.xlabel('M (GeV)')
plt.show()

sns.histplot(df, x='E1', bins=40)
plt.title('Histogram of E1')
plt.xlabel('E1 (GeV)')
plt.show()
