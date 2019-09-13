from frcnn.model import FasterRCNN


def train_frcnn():
    fast = FasterRCNN()
    fast.load_data_train()
    fast.build()
    fast.train()


def test_frcnn():
    fast = FasterRCNN()
    fast.test('./images/')


if __name__ == '__main__':
    test_frcnn()
    


