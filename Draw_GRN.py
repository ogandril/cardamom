# Draw the final GRN
# Remove node with no edges
# Remove_leaves: remove stimulus leaves (Genes with only an input from stimulus and no output)


import os
import numpy as np
import matplotlib.pyplot as plt
from harissa.utils import build_pos, plot_network
from sys import argv 

Time_Line=0
# Time_Line: draw a GRN per time point 

Remove_leaves = 0
# Remove_leaves: remove stimulus leaves (Genes with only an input from stimulus and no output)

D=argv[1]
P=argv[2]
T=argv[3]
cwd=argv[4]

def pgr(datamatrixarray,i):
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
	os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Results")
	ti='GRN.OG'+str(D)+'_'+str(P)+'_'+'Threshold='+str(T)+'_Time='+str(i)+'.pdf'
	#ti='GRN.OG'+str(D)+'_'+str(P)+'_'+'.Threshold='+str(T)+'.pdf'
	fig.savefig(ti, bbox_inches='tight')


os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")
datamatrixarray = np.load('inter.npy')
pgr(datamatrixarray,"a")



if Time_Line:
	os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/cardamom")
	dim=np.load('inter_t.npy')
	for i in range(0, 3):
	#for i in range(0, len(dim)):		    
		datamatrixarray = np.load('inter_{}.npy'.format(i))
		pgr(datamatrixarray)


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
		ti='GRN.OG'+str(D)+'_'+str(P)+'_'+'Threshold='+str(T)+'_Time='+i+'pdf'
		os.chdir(str(cwd)+"/OG"+str(D)+"/"+str(P)+"/Results")
		fig.savefig(ti, bbox_inches='tight')


