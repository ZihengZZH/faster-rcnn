from frcnn.model import FasterRCNN

fast = FasterRCNN()
# fast.load_data_train()
# fast.build()
# fast.train()
fast.test('./images/')
