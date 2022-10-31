import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# CCT Models

__all__ = ['cct_2', 'cct_4', 'cct_6', 'cct_7', 'cct_8', 'cct_14', 'cct_16']


def cct_2(*args, **kwargs):
    return _cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_4(*args, **kwargs):
    return _cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                *args, **kwargs)


def cct_6(*args, **kwargs):
    return _cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_7(*args, **kwargs):
    return _cct(num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_8(*args, **kwargs):
    return _cct(num_layers=8, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cct_14(*args, **kwargs):
    return _cct(num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def cct_16(*args, **kwargs):
    return _cct(num_layers=16, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


def _cct(num_layers, num_heads, mlp_ratio, embedding_dim,
         kernel_size=3, stride=None, padding=None,
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    return CCT(num_layers=num_layers,
               num_heads=num_heads,
               mlp_ratio=mlp_ratio,
               embedding_dim=embedding_dim,
               kernel_size=kernel_size,
               stride=stride,
               padding=padding,
               *args, **kwargs)


# modules

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 attention_dropout=0.1, drop_path_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(dim=d_model, num_heads=nhead,
                                   attention_dropout=attention_dropout, projection_dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Tokenizer_E(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=2,
                 conv_out_size=None,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer_E, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers_E = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding, bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])
        if conv_out_size:
            self.avg_pool = nn.AdaptiveAvgPool2d((conv_out_size[0], conv_out_size[1]))
        else:
            self.avg_pool = nn.Identity()
        self.flattener_E = nn.Flatten(2, 3)

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        e = self.forward(torch.zeros((1, n_channels, height, width)))
        return e.shape[1]

    def forward(self, e):
        return self.flattener_E(self.avg_pool(self.conv_layers_E(e))).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=(3, 3, 1), pooling_stride=(2, 2, 1), pooling_padding=(1, 1, 0),
                 conv_out_size=None,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding, bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool3d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])
        if conv_out_size:
            self.avg_pool = nn.AdaptiveAvgPool3d(conv_out_size)
        else:
            self.avg_pool = nn.Identity()
        self.flattener = nn.Flatten(2, 4)

        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224, depth=12):
        x = self.forward(torch.zeros((1, n_channels, height, width, depth)))
        return x.shape[1]

    def forward(self, x):
        return self.flattener(self.avg_pool(self.conv_layers(x))).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class TransformerClassifier(nn.Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=3,
                 num_heads=4,
                 mlp_ratio=2.0,
                 num_classes=1,
                 dropout_rate=0.1,
                 attention_dropout=0.1,
                 stochastic_depth_rate=0.1,
                 positional_embedding='sine',
                 sequence_length=None,
                 CL_dim=0,
                 *args, **kwargs):
        super().__init__()
        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.CL_dim = CL_dim

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = nn.Parameter(torch.zeros(1, 1, self.embedding_dim),
                                          requires_grad=True)
        else:
            self.attention_pool = nn.Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                   requires_grad=True)
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                   requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward, dropout=dropout_rate,
                                    attention_dropout=attention_dropout, drop_path_rate=dpr[i])
            for i in range(num_layers)])
        self.norm = nn.LayerNorm(embedding_dim)

        self.fc = nn.Linear(embedding_dim, num_classes)
        if CL_dim != 0:
            self.fc_CL = nn.Linear(embedding_dim, CL_dim)
        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.seq_pool:
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x = x[:, 0]
        if self.CL_dim == 0:
            x = self.fc(x)
            return x
        else:
            CL_Vector = self.fc_CL(x)
            x = self.fc(x)
            return x, CL_Vector

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


# CCT Main model
class CCT(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=(5, 5, 1),
                 stride=(5, 5, 1),
                 padding=(2, 2, 0),
                 pooling_kernel_size=(3, 3, 1),
                 pooling_stride=(2, 2, 1),
                 pooling_padding=(1, 1, 0),
                 *args, **kwargs):
        super(CCT, self).__init__()
        img_height, img_width, img_depth = img_size

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_height,
                                                           width=img_width,
                                                           depth=img_depth),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout_rate=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.tokenizer(x)
        return self.out_act(self.classifier(x))


# CCT Main model
class CCT_UDE(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=(5, 5, 1),
                 stride=(5, 5, 1),
                 padding=(2, 2, 0),
                 max_pool=True,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 conv_out_size=None,
                 CL_dim=0,
                 CNN='conv',
                 num_classes=1,
                 *args, **kwargs):
        super(CCT_UDE, self).__init__()
        img_height, img_width, img_depth = img_size
        self.CL_dim = CL_dim
        self.CNN = CNN
        self.embedding_dim = embedding_dim
        if CNN == 'rex50':
            self.tokenizer = torch.hub.load('pytorch/vision:v0.8.2', 'resnext50_32x4d', pretrained=False)
            self.tokenizer.fc = nn.Sequential(nn.Linear(self.tokenizer.fc.in_features, embedding_dim))
            self.tokenizer_E = torch.hub.load('pytorch/vision:v0.8.2', 'resnext50_32x4d', pretrained=False)
            self.tokenizer_E.fc = nn.Sequential(nn.Linear(self.tokenizer_E.fc.in_features, embedding_dim))
            self.classifier = TransformerClassifier(
                sequence_length=13,
                embedding_dim=embedding_dim,
                seq_pool=True,
                dropout_rate=0.,
                attention_dropout=0.1,
                stochastic_depth=0.1,
                CL_dim=CL_dim,
                num_classes=num_classes,
                *args, **kwargs)

        elif CNN == 'res18':
            self.tokenizer = resnet18(pretrained=True)
            self.tokenizer.fc = nn.Sequential(nn.Linear(self.tokenizer.fc.in_features, embedding_dim))
            self.tokenizer_E = resnet18(pretrained=True)
            self.tokenizer_E.fc = nn.Sequential(nn.Linear(self.tokenizer_E.fc.in_features, embedding_dim))
            self.classifier = TransformerClassifier(
                sequence_length=13,
                embedding_dim=embedding_dim,
                seq_pool=True,
                dropout_rate=0.,
                attention_dropout=0.1,
                stochastic_depth=0.1,
                CL_dim=CL_dim,
                num_classes=num_classes,
                *args, **kwargs)
        else:
            self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                       n_output_channels=embedding_dim,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       pooling_kernel_size=pooling_kernel_size,
                                       pooling_stride=pooling_stride,
                                       pooling_padding=pooling_padding,
                                       conv_out_size=conv_out_size,
                                       max_pool=max_pool,
                                       activation=nn.ReLU,
                                       n_conv_layers=n_conv_layers,
                                       conv_bias=False)

            self.tokenizer_E = Tokenizer_E(n_input_channels=n_input_channels,
                                           n_output_channels=embedding_dim,
                                           kernel_size=kernel_size[0],
                                           stride=stride[0],
                                           padding=padding[0],
                                           pooling_kernel_size=pooling_kernel_size[0],
                                           pooling_stride=pooling_stride[0],
                                           pooling_padding=pooling_padding[0],
                                           conv_out_size=conv_out_size,
                                           max_pool=max_pool,
                                           activation=nn.ReLU,
                                           n_conv_layers=n_conv_layers,
                                           conv_bias=False)
            self.classifier = TransformerClassifier(
                sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                               height=img_height,
                                                               width=img_width,
                                                               depth=img_depth) +
                                self.tokenizer_E.sequence_length(n_channels=n_input_channels,
                                                                 height=img_height,
                                                                 width=img_width),
                embedding_dim=embedding_dim,
                seq_pool=True,
                dropout_rate=0.,
                attention_dropout=0.1,
                stochastic_depth=0.1,
                CL_dim=CL_dim,
                num_classes=num_classes,
                *args, **kwargs)

        if num_classes == 1:
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Softmax(dim=1)

    def forward(self, x, e):
        B, C, H, W, T = x.size()
        if self.CNN == 'conv':
            x = self.tokenizer(x)
            e = self.tokenizer_E(e)
        else:
            x = x.permute(0, 4, 1, 2, 3).contiguous()
            x = x.view(B * T, C, H, W)
            x = self.tokenizer(x)
            x = x.view(B, -1, self.embedding_dim)
            e = self.tokenizer_E(e)
            e = torch.unsqueeze(e, 1)
        # print(x.shape, e.shape)
        x = torch.cat([x, e], dim=1)
        if self.CL_dim == 0:
            x = self.classifier(x)
            return self.out_act(x)
        else:
            x, CL_vector = self.classifier(x)
            norm = torch.norm(CL_vector, p='fro', dim=1, keepdim=True)
            return self.out_act(x), CL_vector / norm

'''
momentum_encoder = CCT_UDE(
    img_size=(224, 224, 12),
    embedding_dim=768,
    n_conv_layers=4,
    kernel_size=(7, 7, 1),
    stride=(2, 2, 1),
    padding=(3, 3, 1),
    pooling_kernel_size=(3, 3, 1),
    pooling_stride=(2, 2, 1),
    pooling_padding=(1, 1, 0),
    num_layers=8,
    num_heads=6,
    mlp_radio=2.,
    num_classes=2,
    CL_dim=128,
    positional_embedding='none',  # ['sine', 'learnable', 'none']
)

x = torch.zeros((1, 3, 224, 224, 12))
e = torch.zeros((1, 3, 224, 224))
out = momentum_encoder(x, e)
'''