from transformers import pipeline
from PIL import Image

class Segmenteur:
    def __init__(self):
        self.panoptic_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-panoptic")
        self.segs = None
        self.model = self.panoptic_segmentation.model
        self.labels = self.model.config.id2label
        
    def nouvelle_image(self, image):
        self.segs = self.panoptic_segmentation(image)
        
    def get_mask(self, classe):
        if not classe in self.labels.values():
            return False
        for R in self.segs:
            if R["label"] == classe:
                return R["mask"]