from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """ 此模型用于CycleGAN只从一个方向生成结果"""
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        self.loss_names = []
        self.visual_names = ['real', 'fake']
        self.model_names = ['G' + opt.model_suffix]
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        self.real = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        self.fake = self.netG(self.real)

    def optimize_parameters(self):
        pass
