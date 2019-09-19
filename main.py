import sys
import getopt
from frcnn.model import FasterRCNN


def train_frcnn():
    fast = FasterRCNN()
    fast.load_data_train()
    fast.build()
    fast.train()


def test_frcnn():
    fast = FasterRCNN()
    fast.test('./images/test-images')


def main(argv):
    try:
        opts, _ = getopt.getopt(argv, "re", ["train", "test"])
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ('-r', '--train'):
            train_frcnn()
        elif opt in ('-e', '--test'):
            test_frcnn()
            

if __name__ == '__main__':
    main(sys.argv[1:])
