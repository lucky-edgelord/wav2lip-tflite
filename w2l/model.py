"""Wav2Lip generator, written to load the original Rudrabha/Wav2Lip checkpoints.

Module and attribute names mirror the original so `state_dict` keys line up
(e.g. ``face_encoder_blocks.0.0.conv_block.0.weight``). Only the 4-D input
path is kept: the converter needs fixed shapes, so callers flatten time into
the batch dimension before calling the model.
"""

import torch
from torch import nn


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv_block(x))


def _res(c):
    return Conv2d(c, c, 3, 1, 1, residual=True)


class Wav2Lip(nn.Module):
    """Inputs: mel (B, 1, 80, 16), face (B, 6, 96, 96). Output: (B, 3, 96, 96) in [0, 1]."""

    def __init__(self):
        super().__init__()
        S = nn.Sequential
        self.face_encoder_blocks = nn.ModuleList([
            S(Conv2d(6, 16, 7, 1, 3)),                                        # 96
            S(Conv2d(16, 32, 3, 2, 1), _res(32), _res(32)),                   # 48
            S(Conv2d(32, 64, 3, 2, 1), _res(64), _res(64), _res(64)),         # 24
            S(Conv2d(64, 128, 3, 2, 1), _res(128), _res(128)),                # 12
            S(Conv2d(128, 256, 3, 2, 1), _res(256), _res(256)),               # 6
            S(Conv2d(256, 512, 3, 2, 1), _res(512)),                          # 3
            S(Conv2d(512, 512, 3, 1, 0), Conv2d(512, 512, 1, 1, 0)),          # 1
        ])
        self.audio_encoder = S(
            Conv2d(1, 32, 3, 1, 1), _res(32), _res(32),
            Conv2d(32, 64, 3, (3, 1), 1), _res(64), _res(64),
            Conv2d(64, 128, 3, 3, 1), _res(128), _res(128),
            Conv2d(128, 256, 3, (3, 2), 1), _res(256),
            Conv2d(256, 512, 3, 1, 0), Conv2d(512, 512, 1, 1, 0),
        )
        T = Conv2dTranspose
        self.face_decoder_blocks = nn.ModuleList([
            S(Conv2d(512, 512, 1, 1, 0)),                                     # 1
            S(T(1024, 512, 3, 1, 0), _res(512)),                              # 3
            S(T(1024, 512, 3, 2, 1, 1), _res(512), _res(512)),                # 6
            S(T(768, 384, 3, 2, 1, 1), _res(384), _res(384)),                 # 12
            S(T(512, 256, 3, 2, 1, 1), _res(256), _res(256)),                 # 24
            S(T(320, 128, 3, 2, 1, 1), _res(128), _res(128)),                 # 48
            S(T(160, 64, 3, 2, 1, 1), _res(64), _res(64)),                    # 96
        ])
        self.output_block = S(Conv2d(80, 32, 3, 1, 1), nn.Conv2d(32, 3, 1, 1, 0), nn.Sigmoid())

    def forward(self, mel, face):
        x = self.audio_encoder(mel)
        feats = []
        f = face
        for block in self.face_encoder_blocks:
            f = block(f)
            feats.append(f)
        for block in self.face_decoder_blocks:
            x = block(x)
            x = torch.cat((x, feats.pop()), dim=1)
        return self.output_block(x)


class _CroppedConvTranspose2d(nn.Module):
    """ConvTranspose2d(padding=p, output_padding=op) rewritten as an unpadded
    transposed conv plus a crop. Same weights, same output, but it lowers to a
    plain TRANSPOSE_CONV that the LiteRT converter (and GPU backend) accept.
    """

    def __init__(self, ct: nn.ConvTranspose2d):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ct.in_channels, ct.out_channels, ct.kernel_size,
                                       ct.stride, padding=0, output_padding=0,
                                       bias=ct.bias is not None)
        self.conv.load_state_dict(ct.state_dict())
        self.ph, self.pw = ct.padding
        self.oh = ct.padding[0] - ct.output_padding[0]  # rows to drop at the bottom
        self.ow = ct.padding[1] - ct.output_padding[1]

    def forward(self, x):
        y = self.conv(x)
        h, w = y.shape[-2:]
        return y[:, :, self.ph:h - self.oh, self.pw:w - self.ow]


def make_exportable(model: Wav2Lip) -> Wav2Lip:
    """Fold every BatchNorm into its conv, then rewrite padded ConvTranspose2d
    layers (see _CroppedConvTranspose2d). Call after loading weights."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (Conv2d, Conv2dTranspose)):
            conv, bn = module.conv_block
            _fold_bn(conv, bn)
            module.conv_block[1] = nn.Identity()
            if isinstance(conv, nn.ConvTranspose2d) and (any(conv.padding) or any(conv.output_padding)):
                module.conv_block[0] = _CroppedConvTranspose2d(conv)
    return model


@torch.no_grad()
def _fold_bn(conv, bn):
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    if isinstance(conv, nn.ConvTranspose2d):   # weight: (in, out, kh, kw)
        conv.weight.mul_(scale.view(1, -1, 1, 1))
    else:                                      # weight: (out, in, kh, kw)
        conv.weight.mul_(scale.view(-1, 1, 1, 1))
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    conv.bias = nn.Parameter((bias - bn.running_mean) * scale + bn.bias)


def load_wav2lip(checkpoint_path: str) -> Wav2Lip:
    # weights_only=True refuses arbitrary pickled code, which matters because
    # these checkpoints are usually fetched from third-party mirrors.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model = Wav2Lip()
    model.load_state_dict(state)
    return model.eval()
