# MD-GNN
1. the data used for training and test are in data.csv  
the data is from [pubchem](https://pubchem.ncbi.nlm.nih.gov/),due to the copyright we just list the id of molecules in the file, some methods getting the structures information. descriptors, computing properties and experiment properties can be find in [pubchem site](https://pubchemdocs.ncbi.nlm.nih.gov/programmatic-access)
2. the FF-GNN is in the model ff-gnn.py    
one of the part of mmf-net is the ff-gnn, the correction module is the operation of label which can be realized during training.
3. during training, the data need to organized in the format as shown in [PyG](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data)
