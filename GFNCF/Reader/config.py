
class config():
    def __init__(self):
        # Dataset Parameters
        self.dataset_name = 'ciao_dvd'
        self.rating_path = '../data/%s/ratings.csv' % self.dataset_name
        self.train_path = '../data/%s/split/train.csv' % self.dataset_name
        self.test_path = '../data/%s/split/test.csv' % self.dataset_name
        self.sep = ','
        self.split = 0.8
