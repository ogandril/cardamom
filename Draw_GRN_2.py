# Draw the final GRN

import os
import numpy as np
import matplotlib.pyplot as plt
from harissa.utils import build_pos, plot_network
from sys import argv 

Time_Line=1
# Time_Line: draw a GRN per time point 
# The corresponding matrix have been generated by infer_network.py
# And thresholded by carda.py	

D=argv[1]
P=argv[2]
T=argv[3]
cwd=argv[4]

def pgr(datamatrixarray,i,pos):
	with open('../Data/Genenames.txt') as f:
		Genenames = f.read().splitlines()
		Genenames=np.asarray(Genenames)
# Figure
	fig = plt.figure(figsize=(5,5))
	ax = fig.gca()
# Draw the network
	plot_network(datamatrixarray, pos, axes=ax, names=Genenames, scale=2)
	#plot_network(datamatrixarray, pos, axes=ax, names=Genenames, scale=2, hide_isolated_genes=True, hide_stimulus_leaves=True)
# Export the figure
	os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Results")
	ti='GRN.OG'+str(D)+'_'+str(P)+'_'+'Threshold='+str(T)+'_Time='+str(i)+'.pdf'
	ax.text(-2, 9, ti,fontsize=30)
	fig.savefig(ti, bbox_inches='tight')

# Define pos
os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")
datamatrixarray = np.load('inter.npy')
# Node positions and names
pos = build_pos(datamatrixarray)
# scale to the number of genes
pos *= 8

# Draw complete GRN
pgr(datamatrixarray,"all", pos)

# Draw time-dependent GRNs
if Time_Line:
	os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")
	dim=np.load('inter_t.npy')
	for i in range(0, len(dim)):	
		os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")   
		dmt = np.load('inter_{}.npy'.format(i))
		pgr(dmt,i,pos)


