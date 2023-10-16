# Draw the final GRN

import os
import numpy as np
import matplotlib.pyplot as plt
from harissa.utils import build_pos, plot_network
from sys import argv 

D=argv[1]
P=argv[2]
T=argv[3]
cwd=argv[4]

os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")
#os.chdir("/Users/olivier_2/Documents/En_cours/Labo/Manipes/OG3446/OG3446/1/cardamom")

datamatrixarray = np.loadtxt('inter_matrix.csv', delimiter=',')

# Sum rows
s1=datamatrixarray.sum(axis=0)
# Sum columns
s2=datamatrixarray.sum(axis=1)
# Find when both are null
z1=np.logical_and(s1==0, s2==0)
# Get the index where both are not null
z2=np.where(z1==0)
# Select
datamatrixarray=datamatrixarray[np.ix_(z2[0],z2[0])]

# Get the names of the nodes
with open('../Data/Genenames.txt') as f:
	Genenames = f.read().splitlines()
	g2=np.asarray(Genenames)
	Genenames=g2[np.ix_(z2[0])]

# Node positions and names
pos = build_pos(datamatrixarray)
# scale to the number of genes
pos *= 5

# Figure
fig = plt.figure(figsize=(5,5))
ax = fig.gca()

# Draw the network
plot_network(datamatrixarray, pos, axes=ax, names=Genenames, scale=2)

# Export the figure
ti='GRN.OG'+str(D)+'.Threshold='+str(T)+'.pdf'
os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Results")
fig.savefig(ti, bbox_inches='tight')



