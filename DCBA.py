import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4,dropout=0.0):
        super(FFN, self).__init__()

        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x=self.dropout(x)
        x = self.fc2(x)
        return x

class MyAttention(nn.Module):
    def __init__(self, in_channels1,in_channels2):
        super(MyAttention, self).__init__()
        self.scale = in_channels1 ** -0.5
        self.att_dropout=0

        self.conv1 = nn.Conv2d(in_channels2, in_channels1, kernel_size=3, stride=2, padding=1)
        self.cross_attn1=nn.MultiheadAttention(in_channels1,4,0.2,batch_first=True)
        self.cross_attn2=nn.MultiheadAttention(in_channels1,4,0.2,batch_first=True)


        self.ln1 = nn.LayerNorm(in_channels1)
        self.ln2 = nn.LayerNorm(in_channels1)
        self.ln3 = nn.LayerNorm(in_channels1)
        self.ln4 = nn.LayerNorm(in_channels1)
        self.ln5 = nn.LayerNorm(in_channels1)
        self.ln6 = nn.LayerNorm(in_channels1)

        self.ffn1=FFN(in_channels1,4,0.2)
        self.ffn2=FFN(in_channels1,4,0.2)
        self.ffn3=FFN(in_channels1,4,0.2)


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
        vein=self.conv1(vein)

        original = original.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        vein = vein.permute(0, 2, 3, 1).contiguous().view(b, h * w, c) 

        original_norm=self.ln1(original)
        vein_norm=self.ln2(vein)

        original_out = self.cross_attn1(vein_norm, original_norm, original_norm)[0] + original
        original_out=self.ffn1(self.ln3(original_out))+original_out


        vein_out = self.cross_attn2(original_norm, vein_norm, vein_norm)[0] + vein
        vein_out=self.ffn2(self.ln4(vein_out))+vein_out


        original_out = original_out.view(b, h, w, c).permute(0, 3, 1, 2)  

        vein_out = vein_out.view(b, h, w, c).permute(0, 3, 1, 2) 

        return original_out,vein_out

