import torchvision.transforms as transforms
from util.MEF_SSIM import mef_ssim
from .base_model import BaseModel
from . import networks
import torch


def rescale(x):
    return (x + 1) / 2 * 255


class EvalModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_512', dataset_mode='EVA', netD='n_layers', n_layers_D=6)
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.criterion = mef_ssim
        self.transform = transforms.Grayscale()
        self.netG = networks.define_G(opt.input_nc * 2, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.model_names = ['G']
        self.load_networks(opt.epoch)
        self.image_size = []
        
    def set_input(self, data):
        self.oe = data["oe"].to(self.device)
        self.ue = data["ue"].to(self.device)
        self.cls = data['cls'].to(self.device)
        self.image_paths = data['image_name']
        self.image_size = data['image_size']

    def forward(self):
        # self.gt_gray = self.transform(rescale(self.gt))
        # self.oe_gray = self.transform(rescale(self.oe))
        # self.ue_gray = self.transform(rescale(self.ue))
        # self.img_seq = torch.cat([self.oe_gray, self.ue_gray], 1)
        # self.loss_G_MEFSSIM = self.criterion(self.img_seq, self.gt_gray)
        input_data = torch.cat([self.oe, self.ue], 1)
        d8, d7, d6, d5, d4, e0, e1, e2, e3 = self.netG(input_data, self.cls)
        self.fake_B = d8 
        self.fake_B = self.fake_B * 0.5 + 0.5 # de-normalize

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        self.optimizer.zero_grad()        # set G's gradients to zero
        # self.backward_G()                   # calculate graidents for G
        # self.optimizer.step()             # udpate G's weights
        self.cls.data = torch.clamp(self.cls.data, min=0, max=1)
