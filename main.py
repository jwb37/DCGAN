#!/usr/bin/env python3

from network import Generator, Discriminator

import os
import time
import cv2 as cv
import torch
import torchvision
import torch.nn as nn

import HyperParams as Params


class GAN:
    def __init__(self):
        self.generator = Generator()
        self.generator.to(Params.Device)

        self.discriminator = Discriminator()
        self.discriminator.to(Params.Device)

        self.loss_fn = nn.BCELoss()

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), Params.LearningRateG, betas=(Params.Beta, 0.999))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), Params.LearningRateD, betas=(Params.Beta, 0.999))

        self.exemplar_latent_vectors = torch.randn( (64, Params.LatentVectorSize), device=Params.Device )


    def load_image_set(self):
        transforms = torchvision.transforms.Compose( [
            torchvision.transforms.Resize(Params.ImageSize),
            torchvision.transforms.CenterCrop(Params.ImageSize),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( (0.5,0.5,0.5), (0.5,0.5,0.5) )
        ] )
        self.dataset = torchvision.datasets.ImageFolder(
            Params.ImagePath,
            transforms
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size = Params.BatchSize,
            shuffle=True,
            num_workers=Params.NumWorkers
        )


    def train(self, start_epoch=0):
        self.generator.train()
        self.discriminator.train()

        criterion = nn.BCELoss()

        self.loss_record = {
            'D': [],
            'G': []
        }

        for cur_epoch in range(start_epoch, Params.NumEpochs):
            tic = time.perf_counter()
            
            self.train_epoch(cur_epoch)

            toc = time.perf_counter()
            print( f"Last epoch took: {toc - tic:.0f} seconds" )

            if cur_epoch % Params.CheckpointEvery == 0:
                self.save_checkpoint(f"epoch{cur_epoch+1:03d}")

        self.save_checkpoint(f"Final ({Params.NumEpochs} epochs)")


    def train_epoch(self, cur_epoch):
        for (i, real_images) in enumerate(self.loader):
            # Discriminator training step
            self.discriminator.zero_grad()
            real_images = real_images[0].to(Params.Device)
            batch_size = real_images.size(0)
            labels = torch.full(
                (batch_size,),
                1,
                dtype=torch.float, device=Params.Device
            )
            D_real = self.discriminator(real_images).view(-1)
            D_loss_real = self.loss_fn(D_real, labels)
            D_loss_real.backward()

            latent_vectors = torch.randn( (batch_size, Params.LatentVectorSize), device=Params.Device )
            fake_images = self.generator(latent_vectors)
            D_fake = self.discriminator(fake_images.detach()).view(-1)
            labels.fill_(0)
            D_loss_fake = self.loss_fn(D_fake, labels)
            D_loss_fake.backward()

            self.optimizer_d.step()

            # Generator training step
            self.generator.zero_grad()
            labels.fill_(1)
            D_fake = self.discriminator(fake_images).view(-1)
            G_loss = self.loss_fn(D_fake, labels)
            G_loss.backward()

            self.optimizer_g.step()

            D_loss = D_loss_real.item() + D_loss_fake.item()
            G_loss = G_loss.item()

            if (i % Params.ReportEvery == 0) and i>0:
                print( f"Epoch[{cur_epoch}/{Params.NumEpochs}] Batch[{i}/{len(self.loader)}]" )
                print( f"\tD Loss: {D_loss}\tG Loss: {G_loss}\n" )

            self.loss_record['D'].append(D_loss)
            self.loss_record['G'].append(G_loss)


    def save_checkpoint(self, checkpoint_name):
        dir_path = os.path.join(Params.CheckpointPath, checkpoint_name)
        os.makedirs(dir_path, exist_ok=True)
        print( f"Saving snapshot to: {dir_path}" )

        save_state = dict()
        save_state['g_state'] = self.generator.state_dict()
        save_state['g_optimizer_state'] = self.optimizer_g.state_dict()
        save_state['d_state'] = self.discriminator.state_dict()
        save_state['d_optimizer_state'] = self.optimizer_d.state_dict()

        torch.save(
            save_state,
            os.path.join(dir_path, 'model_params.pt')
        )

        with torch.no_grad():
            self.generator.eval()
            gen_images = self.generator(self.exemplar_latent_vectors)
            self.generator.train()
            torchvision.utils.save_image(
                gen_images,
                os.path.join(dir_path, 'gen_images.png'),
                nrow = 8,
                normalize=True
            )


if __name__ == "__main__":
    g = GAN()
    print( "Loading images" )
    g.load_image_set()
    print( "Beginning training" )
    g.train()
