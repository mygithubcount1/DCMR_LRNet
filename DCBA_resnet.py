import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):

    def __init__(self, dim, mlp_ratio=4, dropout=0.0):
        super(FFN, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)  
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x + residual


class MyAttention(nn.Module):
    def __init__(self, in_channels1, in_channels2, reduction_ratio=4):

        super(MyAttention, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2

        self.reduced_dim = in_channels1 // reduction_ratio

        self.conv1 = nn.Conv2d(in_channels2, in_channels1, kernel_size=3, stride=2, padding=1)

        self.down_conv = nn.Conv2d(in_channels1, self.reduced_dim, kernel_size=1, stride=1, padding=0)  
        self.up_conv = nn.Conv2d(self.reduced_dim, in_channels1, kernel_size=1, stride=1, padding=0)  



        self.scale = self.reduced_dim ** -0.5  
        self.cross_attn1 = nn.MultiheadAttention(self.reduced_dim, 4, 0.2, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(self.reduced_dim, 4, 0.2, batch_first=True)


        self.ln1 = nn.LayerNorm(self.reduced_dim)
        self.ln2 = nn.LayerNorm(self.reduced_dim)
        self.ln3 = nn.LayerNorm(self.reduced_dim)
        self.ln4 = nn.LayerNorm(self.reduced_dim)


        self.ffn1 = FFN(self.reduced_dim, mlp_ratio=4, dropout=0.2) 
        self.ffn2 = FFN(self.reduced_dim, mlp_ratio=4, dropout=0.2)


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, original, vein):
        b, c, h, w = original.size()  
        residual_original = original  

        vein = self.conv1(vein)  

        original_down = self.down_conv(original)  
        vein_down = self.down_conv(vein)  
        reduced_c = original_down.size(1) 

        original_flat = original_down.permute(0, 2, 3, 1).contiguous().view(b, h * w, reduced_c)
        vein_flat = vein_down.permute(0, 2, 3, 1).contiguous().view(b, h * w, reduced_c)

        original_norm = self.ln1(original_flat)
        vein_norm = self.ln2(vein_flat)

        original_out = self.cross_attn1(vein_norm, original_norm, original_norm)[0] + original_flat
        original_out = self.ffn1(original_out)

        vein_out = self.cross_attn2(original_norm, vein_norm, vein_norm)[0] + vein_flat
        vein_out = self.ffn2(vein_out)

        original_out = original_out.view(b, h, w, reduced_c).permute(0, 3, 1, 2)  
        vein_out = vein_out.view(b, h, w, reduced_c).permute(0, 3, 1, 2)  

  
        original_out = self.up_conv(original_out) + residual_original  
        vein_out = self.up_conv(vein_out)

        return original_out, vein_out

