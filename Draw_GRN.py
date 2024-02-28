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

def pgr(datamatrixarray,i):
	with open('../Data/Genenames.txt') as f:
		Genenames = f.read().splitlines()
		Genenames=np.asarray(Genenames)

# Node positions and names
	pos = build_pos(datamatrixarray)
# scale to the number of genes
	pos *= 5
# Figure
	fig = plt.figure(figsize=(5,5))
	ax = fig.gca()
# Draw the network
	plot_network(datamatrixarray, pos, axes=ax, names=Genenames, scale=2, hide_isolated_genes=True, hide_stimulus_leaves=True)
# Export the figure
	os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Results")
	ti='GRN.OG'+str(D)+'_'+str(P)+'_'+'Threshold='+str(T)+'_Time='+str(i)+'.pdf'
	fig.savefig(ti, bbox_inches='tight')


os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")
datamatrixarray = np.load('inter.npy')
pgr(datamatrixarray,"all")


if Time_Line:
	os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")
	dim=np.load('inter_t.npy')
	for i in range(0, len(dim)):	
		os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")	    
		datamatrixarray = np.load('inter_{}.npy'.format(i))
		pgr(datamatrixarray,i)

