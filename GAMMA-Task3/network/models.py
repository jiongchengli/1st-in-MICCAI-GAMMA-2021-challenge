from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet, resnet_cbam, efficientnet, se_resnet, mobilenetv2
import torchvision.models as models


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if backbone_name.endswith('cbam'):
        backbone = resnet_cbam.__dict__[backbone_name](
            replace_stride_with_dilation=replace_stride_with_dilation)
    elif 'resnet34' == backbone_name:
        backbone = models.resnet34(pretrained=pretrained_backbone)
    elif 'se_resnet34' == backbone_name:
        backbone = se_resnet.se_resnet34()
        pretrained_dict = models.resnet34(pretrained=True).state_dict()
        state_dict = backbone.state_dict()
        for k, v in pretrained_dict.items():
            k = k.replace("module.", "")
            if k.startswith('_fc') or k.startswith('fc'):
                continue
            if k in state_dict:
                state_dict[k] = v
        backbone.load_state_dict(state_dict)
    else:
        backbone = resnet.__dict__[backbone_name](
            pretrained=pretrained_backbone,
            replace_stride_with_dilation=replace_stride_with_dilation)

    if 'resnet34' in backbone_name:
        inplanes = 512
        low_level_planes = 64
    else:
        inplanes = 2048
        low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    else:
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_efficientnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    inplanes = 1408
    if pretrained_backbone:
        backbone = efficientnet.EfficientNet.from_pretrained(backbone_name)
    else:
        backbone = efficientnet.EfficientNet.from_name(backbone_name)

    low_level_planes = 256

    if name == 'deeplabv3plus':
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    else:
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name == 'deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    else:
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if 'mobilenet' in backbone:
        model = _segm_mobilenet(arch_type, num_classes, output_stride=output_stride,
                                pretrained_backbone=pretrained_backbone)
    elif 'resnet' in backbone:
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone)
    elif 'efficientnet' in backbone:
        model = _segm_efficientnet(arch_type, backbone, num_classes, output_stride=output_stride,
                                   pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3_resnet34(num_classes=3, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet34', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3_se_resnet34(num_classes=3, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'se_resnet34', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3_efficientnet_b2(num_classes=3, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'efficientnet-b2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet34(num_classes=3, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet34', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_se_resnet34(num_classes=3, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'se_resnet34', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)
