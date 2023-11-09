from Reader.config import config
from collections import defaultdict


class Reader():

    def __init__(self):
        self.config = config()
        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.dataSet_u = defaultdict(dict)
        self.trainSet_u = defaultdict(dict)
        self.trainSet_i = defaultdict(dict)
        self.testSet_u = defaultdict(dict)
        self.testSet_i = defaultdict(dict)
        self.trainSetLength = 0
        self.testSetLength = 0
        self.generate_data_set()
        self.getDataSet()



    def generate_data_set(self):
        for line in self.trainSet():
            u, i, r = line
            if not u in self.user:
                self.user[u] = len(self.user) # user to id
                self.id2user[self.user[u]] = u # id to user
            if not i in self.item:
                self.item[i] = len(self.item) # item to id
                self.id2item[self.item[i]] = i# id to item

            self.trainSet_u[u][i] = r
            self.trainSet_i[i][u] = r
            self.trainSetLength += 1

        for line in self.testSet():
            u, i, r = line
            if u not in self.user or i not in self.item:
                print('>>>>>>>>>>>>error>>>>>>')
            self.testSet_u[u][i] = r
            self.testSet_i[i][u] = r
            self.testSetLength += 1
        pass

    def trainSet(self):
        with open(self.config.train_path, 'r') as f:
            for line in f:
                u, i, r = line.strip('\r\n').split(' ')[0:3]
                yield (int(float(u)), int(float(i)), float(r))

    def testSet(self):
        with open(self.config.test_path, 'r') as f:
            for line in f:
                u, i, r = line.strip('\r\n').split(' ')[0:3]
                yield (int(float(u)), int(float(i)), float(r))

    def getDataSet(self):
        with open(self.config.rating_path, 'r') as f:
            for line in f:
                u, i, r = line.strip('\r\n').split(self.config.sep)[0:3]
                self.dataSet_u[int(u)][int(i)] = float(r)
