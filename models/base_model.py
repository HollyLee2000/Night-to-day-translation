import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """
    这个类是一个用于模型的抽象基类
    若要创建一个子类，您需要实现以下五个函数：
        -- <__init__>:                      初始化该类
        -- <set_input>:                     从数据集中获取数据并预处理。
        -- <forward>:                       产生中间结果
        -- <optimize_parameters>:           计算损失、梯度、更新网络权重。
        -- <modify_commandline_options>:    添加特定于模型的选项，并设置默认选项
    """

    def __init__(self, opt):
        """
        初始化BaseModel类
        参数:
            opt (Option class)-- opt是BaseOptions的子类
        有四个列表需要定义：
            -- self.loss_names:          确定要保存的训练损失
            -- self.model_names:         定义训练网络
            -- self.visual_names:        确定要展示/保存的图片
            -- self.optimizers:          定义并初始化优化器
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # 确定设备是CPU还是GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # 保存checkpoints的目录
        if opt.preprocess != 'scale_width':  # 使用[scale_width]，输入图像可能具有不同的大小，这会影响cudnn.benchmark的性能
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
            添加新的特定于模型的命令参数，并重写现有参数的默认值
            参数:
                parser          -- 命令参数
                is_train (bool) -- 是属于训练的还是测试的
                返回值:
                修改后的命令参数
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """
            从数据集中获取数据并预处理。
            input (dict): 包括数据本身及其元数据信息
        """
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        """计算损失、梯度、更新网络权重，在每次训练迭代中调用 """
        pass

    def setup(self, opt):
        """加载打印网络，创建调度程序 """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)

    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def test(self):
        """测试期间的前传函数"""
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """计算用于visdom可视化输出的图像"""
        pass

    def get_image_paths(self):
        """返回用于加载当前数据的图像路径 """
        return self.image_paths

    def update_learning_rate(self):
        """更新所有网络的学习率；在每个epoch结束时被调用 """
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret

    def save_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
