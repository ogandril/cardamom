import sys; sys.path += ['../']
import numpy as np
from harissa import NetworkModel
import getopt
from alive_progress import alive_bar
from sys import argv 


def build_data(data_real, data_bool, time, my_k, model, basal, inter, transfert, rval):

    C, G = data_real.shape
    my_data = np.zeros((C + 1, G + 1), dtype='uint16')
    my_data[0, 1:] = np.arange(G)
    my_data[1:, 0] = time  # Time points
    my_data[1:, 1] = 1 * (time > 0)  # Stimulus

    # Build the interaction matrix. 
    model.inter[:, :] = inter[:, :]
    model.basal[:] = basal[:]

    C_0 = int(np.sum(data_real[:, 0] == 0))
    c = np.arange(C_0)
    my_data[1:C_0+1, 2:] = data_real[data_real[:, 0] == 0, 1:]
    cnt_k, cnt = 0, 1
    with alive_bar(C) as bar:
        for k in range(C):
            if time[k] > time[k - 1]:
                if my_k[cnt] - C_0 > 0:
                    c = list(np.argwhere(data_real[:, 0] == 0)[:, 0])
                    tmp = C_0
                    while my_k[cnt] - tmp > C_0:
                        c += list(np.argwhere(data_real[:, 0] == 0)[:, 0])
                        tmp += C_0
                    c += list(np.random.choice(np.argwhere(data_real[:, 0] == 0)[:, 0], my_k[cnt] - tmp, replace=False))
                else:
                    c = np.random.choice(np.argwhere(data_real[:, 0] == 0)[:, 0], my_k[cnt], replace=False)
                cnt_k, cnt = 0, cnt + 1
            if k >= C_0:
                sim = model.simulate(time[k], data_real[c[cnt_k], :], data_bool[c[cnt_k], :], use_numba=True)
                my_data[k + 1, 2:] = np.random.poisson(sim.m[-1])
            cnt_k += 1
            bar()

    return my_data


def main(argv):
    inputfile = ''
    transfert = ''
    rval = ''	 
    try:
        opts, args = getopt.getopt(argv,"hi:t:r:",["ifile=","tfile=","rfile="])
        #opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        sys.exit(2)

    #for opt, arg in opts:
    #    if opt in ("-i", "--ifile"):
    #        inputfile = arg

    for opt, arg in opts:
        if opt in ("-i", "--ifile"): 
            inputfile = arg 
        elif opt in ("-t", "--tfile"): 
            transfert = float(arg)
        elif opt in ("-r", "--rfile"):
            rval = float(arg)

    p = '{}/'.format(inputfile)  # Name of the file where are the data

    # Real data
    data_real = np.loadtxt(p+'Data/panel_real.txt', dtype=float, delimiter='\t')[1:, 1:].T
    data_real[:, 0] = np.loadtxt(p+'Data/panel_real.txt', dtype=float, delimiter='\t')[0, 1:]
    G = np.size(data_real, 1) - 1

    # Initialize the model
    model = NetworkModel(G)
    model.a = np.zeros((3, G+1))
    model.a[0, :] = np.load(p + 'cardamom/kmin.npy')
    model.a[1, :] = np.load(p + 'cardamom/kmax.npy')
    model.a[2, :] = np.load(p + 'cardamom/bet.npy')
    data_bool = np.load(p + 'cardamom/data_bool.npy')
    basal = np.load(p + 'cardamom/basal.npy')
    inter = np.load(p + 'cardamom/inter.npy')

    model.d = np.loadtxt(p + 'Rates/degradation_rates.txt', dtype=float, delimiter='\t').T

    # Build the timepoints
    t_real = list(set(data_real[:, 0]))
    t_real.sort()
    t = np.array(t_real, dtype=int)
    #t[-1] *= factor_last
    my_k = [np.sum(data_real[:, 0] == times) for times in t_real]
    C = int(np.sum(my_k))
    k = [int(np.sum(my_k[:i])) for i in range(0, len(t_real))] + [C]
    time = np.zeros(C)
    for i in range(0, len(t)): time[k[i]:k[i + 1]] = t[i]

    ### Build the datasets
    data = build_data(data_real, data_bool, time, my_k, model, basal, inter, transfert, rval)
    fname = p + 'Data/panel_simulated.txt'
    np.savetxt(fname, data.T, fmt='%d', delimiter='\t')

if __name__ == "__main__":
   main(sys.argv[1:])
