import torch
import torch.nn as nn

from stylenet import vgg, decoder


class StyleTransferModule(nn.Module):
    def __init__(self, preserve_colour=True, alpha=1.0, cuda=True):
        super(StyleTransferModule, self).__init__()
        self.cuda = cuda if torch.cuda.is_available() else False
        self.decoder = decoder
        self.vgg = vgg
        self.preserve_colour = preserve_colour
        self.alpha = alpha

        self.decoder.eval()
        self.vgg.eval()

        decoder_path = 'style-models/decoder.pth'
        vgg_path = 'style-models/vgg_normalised.pth'
        device = 'cpu'
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        self.vgg.load_state_dict(torch.load(vgg_path, map_location=device))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])

        if self.cuda:
            self.vgg = self.vgg.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, content, style_in):
        content = content / 255.
        style_in = style_in / 255.

        outputs = None
        for idx in range(content.size(0)):
            image = content[idx, :, :, :]
            style = style_in.squeeze()
            if self.preserve_colour is None:
                preserve_colour = torch.randint(2, (1,)).item()
            else:
                preserve_colour = self.preserve_colour

            with torch.no_grad():
                if preserve_colour:
                    style = coral(style.cpu(), image.cpu())

                if self.cuda:
                    image = image.cuda()
                    style = style.cuda()

                output = self.transfer(image.unsqueeze(0), style.unsqueeze(0))

            output = torch.round(output * 255)
            output[output < 0] = 0
            output[output > 255] = 255
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), dim=0)

            # outputs[idx, :, :, :] = output
            # outputs = outputs.float() / 255.
        return outputs.contiguous()

    def transfer(self, content, style):
        assert (0.0 <= self.alpha <= 1.0)
        content_f = self.vgg(content)
        style_f = self.vgg(style)
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * self.alpha + content_f * (1 - self.alpha)
        return self.decoder(feat)

    # @staticmethod
    # def fix_size(image, h, w):
    #     delta_h = int(round(abs(image.shape[0] - h) / 2))
    #     delta_w = int(round(abs(image.shape[1] - w) // 2))
    #     image = image[:, delta_h:delta_h+h, delta_w:delta_w+w]
    #     return image


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3).cpu()

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3).cpu()

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    #    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())
