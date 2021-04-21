import pytorch_lightning as pl
from models.common import *
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from utils.plot import plot_fields_quad
from datasets.structured import StructuredModule
import yaml

class ConditionalNorm(nn.Module):
    """
    Condition a N*C*H*W query on a N*C latent code.
    """
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.num_features = num_features
        self.IN = nn.InstanceNorm2d(num_features)

    def forward(self, x, c):
        x = self.IN(x)
        return x * c[..., 0, None, None] + c[..., 1, None, None]

class MeshBaseline(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        geo_dim     = kwargs.get('geo_dim', 2)
        param_dim   = kwargs.get('param_dim', 2)
        latent_dim  = kwargs.get('latent_dim', 32)
        out_dim     = kwargs.get('out_dim', 3)

        self.encoder_geo    = CNNEncoder(geo_dim, latent_dim)
        self.encoder_param  = MLPEncoder(param_dim, latent_dim*2)
        self.cNorm          = ConditionalNorm(latent_dim)
        self.res            = nn.Sequential(
            ResNeXt(latent_dim, 16),
            ResNeXt(latent_dim, 16),
        )
        self.decoder        = CNNDecoder(
            method = 'bilinear', groups=4,
            in_channels=latent_dim, out_channels=out_dim,)

    def forward(self, batch):
        [x, c], y = batch
        size = x.size()[-2:]
        # transform
        x = transforms.functional.resize(x, [128, 512])

        zx = self.encoder_geo(F.instance_norm(x))
        zp = self.encoder_param(c).view(c.size()[0], -1, 2)
        out = self.decoder(self.res(self.cNorm(zx, zp)))

        # inverse_transform
        out = transforms.functional.resize(out, size)
        return out
    
    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = 10 * F.l1_loss(out, batch[-1]) + F.mse_loss(out, batch[-1])
        self.log('training_loss', loss)
        if batch_idx == 0:
            [x, _], y = batch
            self.train_example = x[0].detach().cpu(), torch.stack((out[0], y[0], torch.abs(out[0]-y[0]))).detach().cpu() 
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = F.mse_loss(out, batch[-1])
        self.log('val_loss', loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'training_loss',
            'interval': 'step',
            'frequency': 10,
            'strict': True
        }

    def on_train_epoch_end(self, outputs):
        X, fields = self.train_example
        fig = plot_fields_quad(X, fields)
        self.logger.experiment.add_figure(f"Epoch {self.current_epoch}", fig)

if __name__=='__main__':

    cfg_file = './configs/structured.yaml'

    with open(cfg_file, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    
    data_module = StructuredModule(**cfg.get('data'))
    
    model = MeshBaseline(**cfg.get('model'))

    trainer = pl.Trainer(gpus=[0], **cfg.get('training'))

    trainer.fit(model, data_module)
