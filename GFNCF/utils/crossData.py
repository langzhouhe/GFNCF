import pandas as pd
from Reader.config import config
import numpy as np
if __name__ == '__main__':
    conf = config()
    data = pd.read_csv(conf.rating_path, sep=conf.sep, header=None, index_col=None, names=['user', 'item', 'rating'], usecols=[0,1,2],engine='python')
    users = {}
    items = {}
    inds = set()
    # 使得训练集包含所有的user和item
    for indx, line in enumerate(data.values):
        #print(line)
        if int(line[0]) not in users:
            users[int(line[0])] = int(line[0])
            inds.add(indx)
            #continue
        if int(line[1]) not in items:
            items[int(line[1])] = int(line[1])
            inds.add(indx)
            #continue
    data['user'] = data['user'].astype(int)
    data['item'] = data['item'].astype(int)
    data['rating'] = data['rating'].astype(np.float32)

    train_data_1 = data.iloc[list(inds)]
    print(len(train_data_1['user'].unique()))
    print(len(train_data_1['item'].unique()))
    frac = conf.split - len(inds)/len(data)
    data = data.drop(list(inds)).reset_index(drop=True)
    data = data.sample(frac=1)
    train_data_2 = data.iloc[:int(len(data)*frac)]
    train_data = pd.concat([train_data_1, train_data_2], ignore_index=True)
    train_data = train_data.sample(frac=1)
    test_data = data.iloc[int(len(data)*frac)+1:]
    train_path = '../data/%s/split/train.csv' % conf.dataset_name
    test_path = '../data/%s/split/test.csv' % conf.dataset_name
    train_data.to_csv(train_path, header=None, index=None, sep=' ')
    test_data.to_csv(test_path, header=None, index=None, sep=' ')
