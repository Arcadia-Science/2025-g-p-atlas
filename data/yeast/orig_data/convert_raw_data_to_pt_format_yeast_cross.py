import os
import pickle as pk

out_dict={}

phen_file = open('BYxRM_PhenoData.txt' , 'r')

phens = phen_file.read().split('\n')

phens = [x.split('\t') for x in phens]

out_dict['phenotype_names'] = phens[0][1:]

out_dict['strain_names'] = [x[0] for x in phens[1:-1]]

out_dict['phenotypes'] = [x[1:] for x in phens[1:-1]]

out_dict['phenotypes'] = [[float(y)  if y!= 'NA' else 0 for y in x[1:]] for x in phens[1:-1]]

genotype_file = open('BYxRM_GenoData.txt' , 'r')

gens = genotype_file.read().split('\n')

gens = [x.split('\t') for x in gens]

out_dict['loci'] = [x[0] for x in gens[1:-1]]

new_coding_dict = {'R':[0,1],'B':[1,0]}

out_dict['genotypes'] = [[new_coding_dict[x] for x in [gens[y][n] for y in range(len(gens))[1:-1]]] for n in range(len(gens[0]))[1:]]


pk.dump(out_dict, open('yeast_dat_cross_torch_format.pk','wb'))
