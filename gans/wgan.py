import torch
from torch.autograd import Variable
import torchvision.utils as vutils

from tqdm import tqdm
from utils.data import cifar10, fashion_mnist
from utils import TQDMReporter, VisdomReporter
from models import Generator, Discriminator


class Critic(Discriminator):
    def forward(self, input):
        x = self.main(input)
        return x.view(-1)


def main(epochs, z_size, batch_size, output_dir, dataset, sample_batch_size=32, num_update_dis=5):
    assert torch.cuda.is_available(), "This script is only for GPU available environment"
    data_loader, num_color = {"cifar10": (cifar10(batch_size, 64), 3),
                              "fashionmnist": (fashion_mnist(batch_size, 64), 1)}[dataset.lower()]
    torch.backends.cudnn.benchmark = True
    generator = Generator(num_color, 64, z_size)
    critic = Critic(num_color, 64)
    generator.cuda()
    critic.cuda()

    gen_optimizer = torch.optim.RMSprop(params=generator.parameters(), lr=5e-5)
    dis_optimizer = torch.optim.RMSprop(params=critic.parameters(), lr=5e-5)

    fixed_noise = Variable(torch.zeros(sample_batch_size, z_size, 1, 1).normal_(0, 1)).cuda()
    _range = TQDMReporter(range(epochs))
    viz = VisdomReporter(save_dir="log")

    iteration = 0
    for ep in _range:
        for input, _ in tqdm(data_loader, ncols=80):
            iteration += 1
            input = Variable(input).cuda()
            noise = Variable(torch.zeros(batch_size, z_size, 1, 1).normal_(0, 1)).cuda()

            for _ in range(num_update_dis):
                # train discriminator on real data
                dis_optimizer.zero_grad()
                output = critic(input)
                loss_d_r = output.mean()
                loss_d_r.backward()

                # train discriminator on fake data
                output = critic(generator(noise).detach())
                loss_d_f = - output.mean()
                loss_d_f.backward()
                loss_d = loss_d_r + loss_d_f
                dis_optimizer.step()

                # weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # train generator
            gen_optimizer.zero_grad()
            output = critic(generator(noise))
            loss_g = output.mean()
            loss_g.backward()
            gen_optimizer.step()

            if iteration % 10 == 0:
                _range.add_scalars(dict(Gerr=loss_g.data.sum(), Derr=loss_d.data.sum()),
                                   name="errors", idx=iteration)
                viz.add_scalars(dict(Gerr=loss_g.data.sum(), Derr=loss_d.data.sum()),
                                name="errors", idx=iteration)

        gen_image = generator(fixed_noise).data.cpu()
        vutils.save_image(gen_image,
                          f"./{output_dir}/fake_sample_{ep}.png",
                          normalize=True)
        viz.add_images(gen_image, "images", iteration)
    viz.save()


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

    main(500, 100, 64, args.outputdir, args.dataset)
