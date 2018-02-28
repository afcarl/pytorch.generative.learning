import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.utils as vutils

from tqdm import tqdm
from utils.data import cifar10, fashion_mnist
from utils import TQDMReporter, VisdomReporter
from models import Generator, Discriminator


def main(epochs, z_size, batch_size, output_dir, dataset, hyper_parameters):
    assert torch.cuda.is_available(), "This script is only for GPU available environment"
    data_loader, num_color = {"cifar10": (cifar10(batch_size, hyper_parameters.image_size), 3),
                              "fashionmnist": (fashion_mnist(batch_size,
                                                             hyper_parameters.image_size), 1)}[dataset.lower()]
    torch.backends.cudnn.benchmark = True
    generator = Generator(num_color, hyper_parameters.gen_hidden, z_size)
    discriminator = Discriminator(num_color, hyper_parameters.dis_hidden)
    generator.cuda()
    discriminator.cuda()

    gen_optimizer = torch.optim.Adam(params=generator.parameters(),
                                     lr=hyper_parameters.lr, betas=hyper_parameters.betas)
    dis_optimizer = torch.optim.Adam(params=discriminator.parameters(),
                                     lr=hyper_parameters.lr, betas=hyper_parameters.betas)

    # labels
    real_label = Variable(torch.ones(batch_size)).cuda()
    fake_label = Variable(torch.zeros(batch_size)).cuda()
    fixed_noise = Variable(torch.zeros(hyper_parameters.sample_batch_size,
                                       z_size, 1, 1).normal_(0, 1)).cuda()

    iteration = 0
    with TQDMReporter(range(epochs)) as _range, VisdomReporter(save_dir="log") as viz:
        for ep in _range:
            for input, _ in tqdm(data_loader, ncols=80):
                iteration += 1
                input = Variable(input).cuda()
                noise = Variable(torch.zeros(batch_size, z_size, 1, 1).normal_(0, 1)).cuda()

                for _ in range(hyper_parameters.num_update_dis):
                    # train discriminator on real data
                    dis_optimizer.zero_grad()
                    output = discriminator(input)
                    loss_d_r = F.binary_cross_entropy(output, real_label)
                    loss_d_r.backward()

                    # train discriminator on fake data
                    output = discriminator(generator(noise).detach())
                    loss_d_f = F.binary_cross_entropy(output, fake_label)
                    loss_d_f.backward()
                    loss_d = loss_d_r + loss_d_f
                    dis_optimizer.step()

                # train generator

                gen_optimizer.zero_grad()
                output = discriminator(generator(noise))
                # minimizing -log(D(G(z))) instead of maximizing -log(1-D(G(z)))
                loss_g = F.binary_cross_entropy(output, real_label)
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


if __name__ == '__main__':
    import pathlib
    import argparse
    from utils import HyParameter

    hp = HyParameter("configs/gan.yaml")
    p = argparse.ArgumentParser()
    p.add_argument("--outputdir", default="output")
    p.add_argument("--dataset", default="cifar10")
    hp.register_hp(args=p.parse_args())

    path = pathlib.Path(hp.outputdir)
    if not path.exists():
        path.mkdir()
    main(hp.epochs, hp.latent_size, hp.batch_size,
         hp.outputdir, hp.dataset, hp)
