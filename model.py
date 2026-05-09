import timm
import torch
import torch.nn as nn
from DCBA import MyAttention
from thop import profile 


class MainModel(nn.Module):
    def __init__(self, name, pretrained=False,pretrained_cfg_overlay=None):
        super().__init__()

        self.backbone3 = timm.create_model(name,pretrained=pretrained,pretrained_cfg_overlay=pretrained_cfg_overlay)

        self.backbone1 = timm.create_model(name, pretrained=pretrained,pretrained_cfg_overlay=pretrained_cfg_overlay)
        self.backbone1.conv_stem = nn.Conv2d(1, self.backbone1.conv_stem.out_channels, 3, 2, 1, bias=False)

        self.backbone3.conv_head = nn.Identity()
        self.backbone3.bn2 = nn.Identity()
        self.backbone3.global_pool = nn.Identity()
        self.backbone3.classifier = nn.Identity()

        self.backbone1.conv_head = nn.Identity()
        self.backbone1.bn2 = nn.Identity()
        self.backbone1.global_pool = nn.Identity()
        self.backbone1.classifier = nn.Identity()


        self._feat3_b3 = None
        self._feat3_b6 = None
        self._feat1_b3 = None
        self._feat1_b6 = None


        self.backbone3.blocks[3].register_forward_hook(self._get_hook('_feat3_b3'))
        self.backbone3.blocks[6].register_forward_hook(self._get_hook('_feat3_b6'))
        self.backbone1.blocks[3].register_forward_hook(self._get_hook('_feat1_b3'))
        self.backbone1.blocks[6].register_forward_hook(self._get_hook('_feat1_b6'))

    def _get_hook(self, attr_name):
        def hook_fn(module, input, output):
            setattr(self, attr_name, output)

        return hook_fn

    def forward(self, original, vein):

        _ = self.backbone3(original)
        _ = self.backbone1(vein)
        return self._feat3_b3, self._feat1_b3, self._feat3_b6, self._feat1_b6


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.num_classes = args.num_classes
        self.main_model = MainModel('efficientnet_b0',pretrained=True,pretrained_cfg_overlay=dict(
            file=args.weights_path))
        self.attention1 = MyAttention(320, 80)
        self.attention2 = MyAttention(320, 80)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fusion_fc1 = nn.Linear(in_features=320, out_features=260)
        self.fusion_fc2 = nn.Linear(in_features=260, out_features=self.num_classes)
        self.fusion_fc3 = nn.Linear(in_features=320, out_features=260)
        self.fusion_fc4 = nn.Linear(in_features=260, out_features=self.num_classes)
        self.fusion_fc5 = nn.Linear(in_features=320, out_features=260)
        self.fusion_fc6 = nn.Linear(in_features=260, out_features=self.num_classes)
        self.fusion_fc7 = nn.Linear(in_features=320, out_features=260)
        self.fusion_fc8 = nn.Linear(in_features=260, out_features=self.num_classes)

    def forward(self, original, vein):
        original_densblock3, border_densblock3, original_feature, border_feature = self.main_model(original, vein)
        original_feature, border_densblock3 = self.attention1(original_feature, border_densblock3)
        border_feature, original_densblock3 = self.attention2(border_feature, original_densblock3)

        original_densblock3 = self.avgpool(original_densblock3).flatten(1)
        original_densblock3 = self.fusion_fc1(original_densblock3)
        original_densblock3 = self.fusion_fc2(original_densblock3)

        border_densblock3 = self.avgpool(border_densblock3).flatten(1)
        border_densblock3 = self.fusion_fc3(border_densblock3)
        border_densblock3 = self.fusion_fc4(border_densblock3)

        original_feature = self.avgpool(original_feature).flatten(1)
        original_feature = self.fusion_fc5(original_feature)
        original_feature = self.fusion_fc6(original_feature)

        border_feature = self.avgpool(border_feature).flatten(1)
        border_feature = self.fusion_fc7(border_feature)
        border_feature = self.fusion_fc8(border_feature)

        out = original_densblock3 + border_densblock3 + original_feature + border_feature
        return out,original_densblock3,border_densblock3,original_feature,border_feature
    
    

if __name__ == '__main__':
    model = MyModel()
    input_size1 = (3, 448, 448)
    input_size2 = (1, 448, 448)
    input1 = torch.randn(1, *input_size1)
    input2 = torch.randn(1, *input_size2)

    flops, params = profile(model, inputs=(input1, input2), verbose=False)
    gflops = flops / 1e9
    params_m = params / 1e6


    print("=" * 55)
    print(f"GFLOPs: {gflops:.2f}")
    print(f"params: {params_m:.2f} M")
    print("=" * 55)