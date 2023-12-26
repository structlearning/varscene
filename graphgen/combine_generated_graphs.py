import pickle

####### set this ######
graphs_save_path = '/path/to/model/DGMG_uncond-sggen_38/38_2/'
n_sample = 10000

graphs = []
for i in range(n_sample):
	with open(graphs_save_path+'graph'+str(i)+'.dat', 'rb') as f:
		graphs.append(pickle.load(f))

###### set this ######
out_file = 'dgmg_graphs_uncond-sggen_38_2.pkl'
pickle.dump(graphs, open(out_file, 'wb'))
