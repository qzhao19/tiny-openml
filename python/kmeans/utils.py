import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """load dataset
    """
    data = []
    with open(file_path, 'r') as file:
        contents = [line.strip().split('\t') for line in file.readlines()]
    
    for i in range(len(contents)):
        data.append(contents[i][:]
        
    return np.array(data).astype(float)           

def show_cluster(data, k, centroids, cluster_assment):  
    n_samples, dim = data.shape  
    if dim != 2:  
        print("Sorry! I can not draw because the dimension of your data is not 2!")
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']  
    if k > len(mark):  
        print("Sorry! Your k is too large! please contact Zouxy")
    # draw all samples  
    for i in range(n_samples):  
        mark_index = int(cluster_assment[i, 0])  
        plt.plot(data[i, 0], data[i, 1], mark[mark_index])  
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']  
    # draw the centroids  
    for i in range(k):  
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize = 12)  
    plt.show()  
