def load_data(file_path):
    """load dataset
    """
    data = []
    with open(file_path, 'r') as file:
        contents = [line.strip().split('\t') for line in file.readlines()]
    
    for i in range(len(contents)):
        data.append(contents[i][:]
        
    return np.array(data).astype(float)
    
