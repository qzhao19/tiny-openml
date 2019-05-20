def downloadData(file_name):
    """Download dating data to testing alogrithm
       Paramters:
           file_name: input file path which allows to read a txt file
       Returns:
           retuen_mat: a matrix of dating data contains 3 attributs: 
                           1. Number of frequent flyer miles earned per year
                           2. Percentage of time spent playing video games
                           3. Liters of ice cream consumed per week
           label_vect: a vectro conatins labels 
    """
    with open(file_name) as file:
        arr_lines=file.readlines()
        num_lines=len(arr_lines)
        return_mat=np.zeros((num_lines, 3), dtype=float)
        label_vect=[]
        index=0
        for line in arr_lines:
            line=line.strip().split('\t')
            return_mat[index,:]=line[0:3]
            label_vect.append(int(line[-1]))
            index+=1
    return return_mat, label_vect


def plot2D(data, labels):
    fig = plt.figure()
    colors = ('red', 'green', 'blue')
    groups = ('Did Not Like', 'Liked in Small Doses', 'Liked in Large Doses') 
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    ax = fig.add_subplot(111)
    for color, group in zip(colors, groups):
        ax.scatter(data[:,0], data[:,1], 
                   15.0*np.array(labels), 15.0*np.array(labels),
                   alpha=0.8, label=group)
    ax.legend(loc=2)
    plt.show()
    