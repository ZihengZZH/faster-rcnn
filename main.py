from frcnn.model import FasterRCNN

fast = FasterRCNN()
fast.load_data()
fast.build()
fast.train()