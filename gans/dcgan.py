import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.utils as vutils

from tqdm import tqdm, trange
from data import cifar10, fashion_mnist


def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(0.0, 0.02)
    elif "BatchNorm" in classname:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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
                nn.Tanh()
                # state size. (nc) x 64 x 64
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)


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
                nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)



def main(epochs, z_size, batch_size, output_dir, dataset):
    assert torch.cuda.is_available(), "This script is only for GPU available environment"
    data_loader = {"cifar10": cifar10(batch_size, 64),
                   "fashionmnist": fashion_mnist(batch_size, 64)}[dataset.lower()]

    generator = Generator(3, 64, z_size)
    discriminator = Discriminator(3, 64)
    generator.cuda()
    discriminator.cuda()

    gen_optimizer = torch.optim.Adam(params=generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    dis_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # labels
    real_label = Variable(torch.ones(batch_size)).cuda()
    fake_label = Variable(torch.zeros(batch_size)).cuda()
    fixed_noise = Variable(torch.zeros(batch_size, z_size, 1, 1).normal_(0, 1)).cuda()
    _range = trange(epochs, ncols=80)

    for ep in _range:
        for input, _ in tqdm(data_loader, ncols=80):
            input = Variable(input).cuda()
            noise = Variable(torch.zeros(batch_size, z_size, 1, 1).normal_(0, 1)).cuda()

            for _ in range(5):
                # train discriminator on real data
                dis_optimizer.zero_grad()
                output = discriminator(input).view(-1)
                loss_d_r = F.binary_cross_entropy(output, real_label)
                loss_d_r.backward()

                # train discriminator on fake data
                output = discriminator(generator(noise).detach()).view(-1)
                loss_d_f = F.binary_cross_entropy(output, fake_label)
                loss_d_f.backward()
                loss_d = loss_d_r + loss_d_f
                dis_optimizer.step()

            # train generator

            gen_optimizer.zero_grad()
            output = discriminator(generator(noise)).view(-1)
            loss_g = F.binary_cross_entropy(output, real_label)
            loss_g.backward()

            gen_optimizer.step()

            _range.set_postfix(Gerr=loss_g.data.sum(), Derr=loss_d.data.sum())

        vutils.save_image(generator(fixed_noise).data.cpu(),
                          f"./{output_dir}/fake_sample_{ep}.png",
                          normalize=True)


if __name__ == '__main__':
    import pathlib
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--outputdir", default="output")
    p.add_argument("--dataset", default="cifar10")
    args = p.parse_args()

    path = pathlib.Path(args.outputdir)
    if not path.exists():
        path.mkdir()

    main(500, 100, 128, args.outputdir, args.dataset)
