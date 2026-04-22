import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights
import copy

from DCBA_resnet import MyAttention



def modify_resnet_first_conv(resnet_model, in_channels=1):
    original_conv = resnet_model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None
    )

    if in_channels == 1:
        new_conv.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
    else:
        new_conv.weight.data = original_conv.weight.data

    resnet_model.conv1 = new_conv
    return resnet_model


class MainModel(nn.Module):
    def __init__(self, backbone_type='resnet50', pretrained=True, pretrained_path=None, device=None):
        super(MainModel, self).__init__()
        self.backbone_type = backbone_type
        self.device = device if device else torch.device('cpu')
        print("Backbone type:", backbone_type)

        if backbone_type.startswith('resnet'):
            if backbone_type == 'resnet50':
                # weights = ResNet50_Weights.DEFAULT if pretrained else None
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                self.backbone_template = models.resnet50(weights=weights)

            if pretrained_path and pretrained:
                state_dict = torch.load(pretrained_path, map_location=self.device)
                self.backbone_template.load_state_dict(state_dict)

            self.model3 = copy.deepcopy(self.backbone_template).to(self.device)
            self.model1 = copy.deepcopy(self.backbone_template).to(self.device)


            self.model1 = modify_resnet_first_conv(self.model1, in_channels=1)

            self._setup_resnet_feature_stages()
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

    def _setup_resnet_feature_stages(self):
        self.model3_densblock1 = nn.Sequential(
            self.model3.conv1,
            self.model3.bn1,
            self.model3.relu,
            self.model3.maxpool,
            self.model3.layer1,
            self.model3.layer2,
            self.model3.layer3
        ).to(self.device)

        self.model3_feature = nn.Sequential(
            self.model3.layer4
        ).to(self.device)

        self.model1_densblock1 = nn.Sequential(
            self.model1.conv1,
            self.model1.bn1,
            self.model1.relu,
            self.model1.maxpool,
            self.model1.layer1,
            self.model1.layer2,
            self.model1.layer3
        ).to(self.device)

        self.model1_feature = nn.Sequential(
            self.model1.layer4
        ).to(self.device)

        if self.backbone_type in ['resnet50']:
            self.stage1_dim = 1024  
            self.feat_dim = 2048  

        self.stage1_size = (14, 14)
        self.feat_size = (7, 7)

    def forward(self, original, vein):
        original = original.to(self.device)
        vein = vein.to(self.device)

        model3_densblock1 = self.model3_densblock1(original)
        model1_densblock1 = self.model1_densblock1(vein)
        model3_feature = self.model3_feature(model3_densblock1)
        model1_feature = self.model1_feature(model1_densblock1)

        return model3_densblock1, model1_densblock1, model3_feature, model1_feature


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.num_classes = args.num_classes
        self.device = args.device

        backbone_type = getattr(args, 'backbone', 'resnet50')
        pretrained_path = getattr(args, 'pretrained_path', None)

        self.main_model = MainModel(
            backbone_type=backbone_type,
            pretrained=True,
            pretrained_path=pretrained_path,
            device=self.device
        )

        self.stage1_dim = self.main_model.stage1_dim
        self.feat_dim = self.main_model.feat_dim
        self.stage1_size = self.main_model.stage1_size
        self.feat_size = self.main_model.feat_size

        if args.freeze_layers:
            self._freeze_backbone_layers()

        self.attention1 = MyAttention(self.feat_dim, self.stage1_dim, reduction_ratio=8).to(self.device)
        self.attention2 = MyAttention(self.feat_dim, self.stage1_dim, reduction_ratio=8).to(self.device)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1).to(self.device)

        self.fusion_fc1 = nn.Linear(in_features=self.feat_dim, out_features=260)
        self.fusion_fc2 = nn.Linear(in_features=260, out_features=args.num_classes)

        self.fusion_fc3 = nn.Linear(in_features=self.feat_dim, out_features=260)
        self.fusion_fc4 = nn.Linear(in_features=260, out_features=args.num_classes)

        self.fusion_fc5 = nn.Linear(in_features=self.feat_dim, out_features=260)
        self.fusion_fc6 = nn.Linear(in_features=260, out_features=args.num_classes)

        self.fusion_fc7 = nn.Linear(in_features=self.feat_dim, out_features=260)
        self.fusion_fc8 = nn.Linear(in_features=260, out_features=args.num_classes)

    def _freeze_backbone_layers(self):
        for name, para in self.main_model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                para.requires_grad_(False)

    def forward(self, original, vein):

        original_densblock3, border_densblock3, original_feature, border_feature = self.main_model(original, vein)

        original_feature, border_densblock3 = self.attention1(original_feature, border_densblock3)
        border_feature, original_densblock3 = self.attention2(border_feature, original_densblock3)

        original_densblock3 = self.avgpool(original_densblock3).view(original_densblock3.size(0), -1)
        original_densblock3 = self.fusion_fc1(original_densblock3)
        original_densblock3 = self.fusion_fc2(original_densblock3)

        border_densblock3 = self.avgpool(border_densblock3).view(border_densblock3.size(0), -1)
        border_densblock3 = self.fusion_fc3(border_densblock3)
        border_densblock3 = self.fusion_fc4(border_densblock3)

        original_feature = self.avgpool(original_feature).view(original_feature.size(0), -1)
        original_feature = self.fusion_fc5(original_feature)
        original_feature = self.fusion_fc6(original_feature)

        border_feature = self.avgpool(border_feature).view(border_feature.size(0), -1)
        border_feature = self.fusion_fc7(border_feature)
        border_feature = self.fusion_fc8(border_feature)

        out = original_densblock3 + border_densblock3 + original_feature + border_feature

        return out, original_densblock3, border_densblock3, original_feature, border_feature
