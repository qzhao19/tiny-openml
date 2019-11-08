import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name):
    """Download dating data to testing alogrithm
    Args:
        file_name: input file path which allows to read a txt file
    Returns:
        retuen_mat: a matrix of dating data contains 3 attributs: 
                        1. Number of frequent flyer miles earned per year
                        2. Percentage of time spent playing video games
                        3. Liters of ice cream consumed per week
        label_vect: a vectro conatins labels 
    """
    with open(file_name, 'r') as file:
        contents = [line.strip().split('\t') for line in file.readlines()]

    label_map = {'largeDoses': '3', 'smallDoses': '2', 'didntLike': '1'}
    feature_list = []
    label_list = []
    for i in range(len(contents)):
        feature_list.append(contents[i][:3])
        label_list.append(contents[i][-1])
    
    for indices, value in enumerate(label_list):
        if not value.isdigit():
            if value in label_map.keys():
                label_list[indices] = label_map[value]
                
    return np.array(feature_list).astype(float), np.asarray(label_list).astype(int)


def plot2D(data, label):
    fig = plt.figure(figsize=(8, 6))
    colors = ('red', 'green', 'blue')
    groups = ('Did Not Like', 'Liked in Small Doses', 'Liked in Large Doses') 
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    ax = fig.add_subplot(111)
    for color, group in zip(colors, groups):
        ax.scatter(data[:,0], data[:,1], 
                   15.0*label, 15.0*label,
                   alpha=0.8, label=group)
    ax.legend(loc=2)
    plt.show()
# plot2D(features, labels)
    
