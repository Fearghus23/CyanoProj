# model.py
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def create_model(num_classes):
    config = get_efficientdet_config('tf_efficientdet_d0')
    net = EfficientDet(config, pretrained_backbone=True)
    config.num_classes = num_classes
    config.image_size = (512, 512)
    net.class_net = HeadNet(config, num_outputs=config.num_classes)
    model = DetBenchTrain(net, config)
    return model
