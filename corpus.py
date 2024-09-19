import os
import shutil

file_path = '/home/mislam/maminur/nsyss/Cisco_22_networks/dir_g21_small_workload_with_gt/dir_includes_packets_and_other_nodes/'
new_dir = '/home/mislam/WalkLM/data/Cisco_22_networks/dir_g21_small_workload_with_gt/dir_includes_packets_and_other_nodes/' + 'corpus/' 
if os.path.exists(new_dir):
	shutil.rmtree(new_dir, ignore_errors=True)
os.makedirs(new_dir)

files = os.listdir(file_path)

for f in files:
	if f.endswith('.gz'):
		continue

	if f.endswith('corpus'):
		continue

	nodes = {}
	file = open(file_path + f)
	s = ''
	for line in file:
		line = line.split('	')
		node1,node2,ps = line[1],line[2],line[3]
		if node1 not in nodes:
			nodes[node1] = 'node_'+ node1
		if node2 not in nodes:
			nodes[node2] = 'node_'+ node2

		s += 'node_' + node1 + ' is communicating with node_' + node2 + ' on '
		ps = ps.split(',')
		for p in ps:
			p = p.split('p')
			p[1] = p[1].split('-')
			s += ' protocol_' + p[0] + ' '
			s += ' with port_' + p[1][0]
			if len(p[1]) > 1:
				s += ' and port_' + p[1][1]



	wfile = open(new_dir + f, "w") 
	wfile.write(s)
	wfile.close()

	s = ''
	for nid in nodes:
		s += nid + '\t' + nodes[nid] + '\n'
	fname = new_dir + f
	fname = fname.replace('.txt','.dat')
	wfile = open(fname, "w") 
	wfile.write(s)
	wfile.close()


		
