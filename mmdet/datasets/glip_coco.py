from .coco import CocoDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class GLIPCocoDataset(CocoDataset):
    def prepare_data(self, idx):
        data = super().prepare_data(idx)
        data['text'] = '. '.join(self.CLASSES)
        data['custom_entities'] = list(self.CLASSES)
        return data
