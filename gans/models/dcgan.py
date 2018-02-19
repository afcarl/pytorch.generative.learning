from torch import nn
from torch.nn import functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal(m.weight, 0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal(m.weight, 1, 0.02)
        nn.init.constant(m.bias, 0)


class Generator(nn.Module):
    def __init__(self, color, gen_hidden, z_size):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(z_size, gen_hidden * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(gen_hidden * 8),
                nn.ReLU(True),
                # state size. (gen_hidden*8) x 4 x 4
                nn.ConvTranspose2d(gen_hidden * 8, gen_hidden * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(gen_hidden * 4),
                nn.ReLU(True),
                # state size. (gen_hidden*4) x 8 x 8
                nn.ConvTranspose2d(gen_hidden * 4, gen_hidden * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(gen_hidden * 2),
                nn.ReLU(True),
                # state size. (gen_hidden*2) x 16 x 16
                nn.ConvTranspose2d(gen_hidden * 2, gen_hidden, 4, 2, 1, bias=False),
                nn.BatchNorm2d(gen_hidden),
                nn.ReLU(True),
                # state size. (gen_hidden) x 32 x 32
                nn.ConvTranspose2d(gen_hidden, color, 4, 2, 1, bias=False),
                # state size. (nc) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, input):
        x = self.main(input)
        return F.tanh(input)


class Discriminator(nn.Module):
    def __init__(self, color, dis_hidden):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(color, dis_hidden, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (dis_hidden) x 32 x 32
                nn.Conv2d(dis_hidden, dis_hidden * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dis_hidden * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (dis_hidden*2) x 16 x 16
                nn.Conv2d(dis_hidden * 2, dis_hidden * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dis_hidden * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (dis_hidden*4) x 8 x 8
                nn.Conv2d(dis_hidden * 4, dis_hidden * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dis_hidden * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (dis_hidden*8) x 4 x 4
                nn.Conv2d(dis_hidden * 8, 1, 4, 1, 0, bias=False),
        )
        self.apply(weights_init)

    def forward(self, input):
        x = self.main(input).view(-1)
        return F.sigmoid(x)
