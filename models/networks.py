import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """返回一个标准化层，默认使用实例归一化，有利于风格迁移，批归一化对小批量的实现效果差"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """返回一个学习率调度
        optimizer          -- 网络优化器
        opt (option class) -- opt是BaseOptions的子类．　
    """
    if opt.lr_policy == 'linear':  # 线性优化
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':  # 其他的一些优化方法，我们没有用到
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络权重
        net (network)     -- 需要被初始化的网络
        init_type (str)   -- 初始化权重方法: normal | xavier | kaiming | orthogonal
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # 批归一化权值不是一个矩阵；只是正态分布的应用。
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """初始化网络，包括设备和权重
    参数:
        net (network)      -- 需要被初始化的网络
        init_type (str)    -- 初始化权重方法: normal | xavier | kaiming | orthogonal
        gain (float)       -- 比例系数
        gpu_ids (int list) -- GPU使用
    返回一个初始化的网络
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """
    制造一个生成器
    参数:
        input_nc (int) -- 输入图像的通道数
        output_nc (int) -- 输出图像的通道数
        ngf (int) -- 最后一个卷积层的卷积核数
        netG (str) -- 网络结构名称: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- 使用的归一化层类型
        use_dropout (bool) -- 是否使用dropout层，一般不使用，因为一般不会过拟合.
        init_type (str)    -- 初始网络权重的方法
        init_gain (float)  -- 部分方法的比例因子
        gpu_ids (int list) -- GPU使用
    返回一个生成器
    生成器网络是我们所求的目标和技术核心，其中配对翻译采用了U-net，非配对采用残差网络
    U-Net原论文: https://arxiv.org/abs/1505.04597
    残差网络生成器代码来自李飞飞和Justin Johnson教授：https://github.com/jcjohnson/fast-neural-style.
    该生成器已进行了初始化，使用relu来增加非线性
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
        制造一个鉴别器
        参数:
            input_nc (int)      -- 输入图像的通道数
            ngf (int)           -- 最后一个卷积层的卷积核数
            netD (str)          -- 网络结构名称: basic | n_layers | pixel
            n_layers_D (int)    -- 鉴别器中的卷积层数
            norm (str)          -- 使用的归一化层类型
            init_type (str)     -- 初始网络权重的方法
            init_gain (float)   -- 部分方法的比例因子
            gpu_ids (int list)  -- GPU使用
        返回一个鉴别器

    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    elif netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


class GANLoss(nn.Module):
    """
    定义对抗性损失
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """
        参数:
            gan_mode (str) - - GAN目标函数种类，通常使用lsgan(最小二乘),会使模型更加稳定
            target_real_label (bool) - - 真图像标签
            target_fake_label (bool) - - 假图像标签
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """
        创造和输入大小一样的标签张量
        参数:
            prediction (tensor)     - - 鉴别器的预测
            target_is_real (bool)   - - 目标标签是否为真
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """
        计算对抗性损失
        参数:
            prediction (tensor)     - - 鉴别器的预测
            target_is_real (bool)   - - 目标标签是否为真
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)  # 如果是lsgan就是MSEloss，如果是vanilla就是BCEWithLogitsLoss
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """
    计算梯度惩罚损失，参考以下论文中的方法：https://arxiv.org/abs/1704.00028
    参数:
        netD (network)              -- 鉴别器网络
        real_data (tensor array)    -- 真图像tensor
        fake_data (tensor array)    -- 生成器输出的假图像tensor
        device (str)                -- GPU / CPU
        type (str)                  -- 使用真图像、假图像还是两者的线性插值的 [real | fake | mixed].
        constant (float)            -- 公式中的常量 ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- 此损失的权重
    """
    if lambda_gp > 0.0:
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        return gradient_penalty, gradients
    else:
        return 0.0, None


class ResnetGenerator(nn.Module):
    """
    残差网络生成器，代码来自李飞飞、Justin Johnson等人的实时风格转换和超分辨项目
    开源代码：https://github.com/jcjohnson/fast-neural-style
    论文：https://cs.stanford.edu/people/jcjohns/eccv16/
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """
        构建一个残差网络生成器
        参数:
            input_nc (int)      -- 输入图像的通道数
            output_nc (int)     -- 输出图像的通道数
            ngf (int)           -- 最后一个卷积层的卷积核数
            norm_layer          -- 归一化层
            use_dropout (bool)  -- 是否使用dropout层
            n_blocks (int)      -- 残差块的个数
            padding_type (str)  -- 卷积层的填充方式: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:  # 是否使用偏差
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        """
        先使用反射填充，用7x7大卷积核进行一次有效卷积，使得尺寸不变，通道数增加为ngf，可以增加图像边缘利用率
        """
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        """
        两次下采样，使用3x3卷积核二跨步卷积，每次通道数翻倍，随后是归一层和relu层。
        归一层我们使用实例归一，参考论文：https://arxiv.org/abs/1607.08022
        """
        n_downsampling = 2
        for i in range(n_downsampling):  # 添加下采样
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # 添加残差块

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]
        """
        两次上采样至所需的分辨率空间
        """
        for i in range(n_downsampling):  # 添加上采样
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        """
        重复第一次大卷积核卷积操作，生成输出
        """
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        构建残差网络，参考论文：https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        构建卷积块
        参数:
            dim (int)           -- 卷积层通道数
            padding_type (str)  -- 填充方式
            norm_layer          -- 归一层
            use_dropout (bool)  -- 是否使用dropout层
            use_bias (bool)     -- 卷积层是否使用偏置
        """
        conv_block = []  # 先padding
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        # 加等维卷积，通道数不变
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        # 重复一遍上述结构
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """残差网络的前传跳跃结构"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """构建Unet生成器，这是一个递归的过程"""
    # 可以改为实例归一，或许效果会更好
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """
        参数:
            input_nc (int)  -- 输入图像的通道数
            output_nc (int) -- 输出图像的通道数
            num_downs (int) -- UNet中的下采样层数
            ngf (int)       -- 卷积核初始数
            norm_layer      -- 归一化层
        """
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # 先添加最内层模块
        for i in range(num_downs - 5):  # 添加中间层模块中过滤器不变的
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # 减少过滤器，靠近最外层模块
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # 添加最外层模块

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """
    定义U-net的跳跃结构
    """
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        # 可以改为实例归一，或许效果会更好
        """
        参数:
            outer_nc (int) -- 上采样方卷积层的卷积核数
            inner_nc (int) -- 其对应的下采样方卷积层的卷积核数
            input_nc (int) -- 输入图像/特征图的通道数
            submodule (UnetSkipConnectionBlock) -- 之前定义的子模块(递归)
            outermost (bool)    -- 是否是最外层模块
            innermost (bool)    -- 是否是最内层模块
            norm_layer          -- 归一化层
            use_dropout (bool)  -- 是否使用dropout层
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:  # 最外层模块结构
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:  # 最内层模块(上采样通道数不变)
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:  # 中间模块
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # 增加跳跃结构
            return torch.cat([x, self.model(x)], 1)


class NLayerDiscriminator(nn.Module):
    """定义一个局部鉴别器"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        参数:
            input_nc (int)  -- 输入图像通道数
            ndf (int)       -- 初始卷积核数量
            n_layers (int)  -- 鉴别器中的卷积层数量
            norm_layer      -- 归一化层
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 下采样，每次通道数翻倍
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # 输出一个单通道预测图，用于计算对抗性损失
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """定义一个像素鉴别器，提高局部约束和清晰度，但失去了大量风格学习能力"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """
        参数:
            input_nc (int)  -- 输入图像通道数
            ndf (int)       -- 初始卷积核数量
            n_layers (int)  -- 鉴别器中的卷积层数量
            norm_layer      -- 归一化层
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
        # 最后直接输出一个预测概率，用于计算对抗性损失
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

# 弃用的模型
"""
class Unet_resize_conv(nn.Module):
    def __init__(self, opt, skip):
        super(Unet_resize_conv, self).__init__()

        self.opt = opt
        self.skip = skip
        p = 1
        # self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
        if opt.self_attention:
            self.conv1_1 = nn.Conv2d(4, 32, 3, padding=p)
            # self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
            self.downsample_1 = nn.MaxPool2d(2)
            self.downsample_2 = nn.MaxPool2d(2)
            self.downsample_3 = nn.MaxPool2d(2)
            self.downsample_4 = nn.MaxPool2d(2)
        else:
            self.conv1_1 = nn.Conv2d(3, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn1_2 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.max_pool1 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn2_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.max_pool2 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn3_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.max_pool3 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn4_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.max_pool4 = nn.AvgPool2d(2) if self.opt.use_avgpool == 1 else nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_1 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn5_2 = SynBN2d(512) if self.opt.syn_norm else nn.BatchNorm2d(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_1 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn6_2 = SynBN2d(256) if self.opt.syn_norm else nn.BatchNorm2d(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_1 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn7_2 = SynBN2d(128) if self.opt.syn_norm else nn.BatchNorm2d(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_1 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn8_2 = SynBN2d(64) if self.opt.syn_norm else nn.BatchNorm2d(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        if self.opt.use_norm == 1:
            self.bn9_1 = SynBN2d(32) if self.opt.syn_norm else nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)
        if self.opt.tanh:
            self.tanh = nn.Tanh()

    def depth_to_space(self, input, block_size):
        block_size_sq = block_size * block_size
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / block_size_sq)
        s_width = int(d_width * block_size)
        s_height = int(d_height * block_size)
        t_1 = output.resize(batch_size, d_height, d_width, block_size_sq, s_depth)
        spl = t_1.split(block_size, 3)
        stack = [t_t.resize(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).resize(batch_size, s_height, s_width,
                                                                                     s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

    def forward(self, input, gray):
        flag = 0
        if input.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            input = avg(input)
            gray = avg(gray)
            flag = 1
            # pass
        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        gray, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(gray)
        if self.opt.self_attention:
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)
        if self.opt.use_norm == 1:
            if self.opt.self_attention:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1))))
                # x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            else:
                x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
            conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
            x = self.max_pool1(conv1)

            x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
            conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
            x = self.max_pool2(conv2)

            x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
            conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
            x = self.max_pool3(conv3)

            x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
            conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
            x = self.max_pool4(conv4)

            x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
            x = x * gray_5 if self.opt.self_attention else x
            conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4 * gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
            conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3 * gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
            conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2 * gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
            conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1 * gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent * gray

            # output = self.depth_to_space(conv10, 2)
            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
                    input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
                    output = latent + input * self.opt.skip
                    output = output * 2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
                    output = latent + input * self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output / torch.max(torch.abs(output))


        elif self.opt.use_norm == 0:
            if self.opt.self_attention:
                x = self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1)))
            else:
                x = self.LReLU1_1(self.conv1_1(input))
            conv1 = self.LReLU1_2(self.conv1_2(x))
            x = self.max_pool1(conv1)

            x = self.LReLU2_1(self.conv2_1(x))
            conv2 = self.LReLU2_2(self.conv2_2(x))
            x = self.max_pool2(conv2)

            x = self.LReLU3_1(self.conv3_1(x))
            conv3 = self.LReLU3_2(self.conv3_2(x))
            x = self.max_pool3(conv3)

            x = self.LReLU4_1(self.conv4_1(x))
            conv4 = self.LReLU4_2(self.conv4_2(x))
            x = self.max_pool4(conv4)

            x = self.LReLU5_1(self.conv5_1(x))
            x = x * gray_5 if self.opt.self_attention else x
            conv5 = self.LReLU5_2(self.conv5_2(x))

            conv5 = F.upsample(conv5, scale_factor=2, mode='bilinear')
            conv4 = conv4 * gray_4 if self.opt.self_attention else conv4
            up6 = torch.cat([self.deconv5(conv5), conv4], 1)
            x = self.LReLU6_1(self.conv6_1(up6))
            conv6 = self.LReLU6_2(self.conv6_2(x))

            conv6 = F.upsample(conv6, scale_factor=2, mode='bilinear')
            conv3 = conv3 * gray_3 if self.opt.self_attention else conv3
            up7 = torch.cat([self.deconv6(conv6), conv3], 1)
            x = self.LReLU7_1(self.conv7_1(up7))
            conv7 = self.LReLU7_2(self.conv7_2(x))

            conv7 = F.upsample(conv7, scale_factor=2, mode='bilinear')
            conv2 = conv2 * gray_2 if self.opt.self_attention else conv2
            up8 = torch.cat([self.deconv7(conv7), conv2], 1)
            x = self.LReLU8_1(self.conv8_1(up8))
            conv8 = self.LReLU8_2(self.conv8_2(x))

            conv8 = F.upsample(conv8, scale_factor=2, mode='bilinear')
            conv1 = conv1 * gray if self.opt.self_attention else conv1
            up9 = torch.cat([self.deconv8(conv8), conv1], 1)
            x = self.LReLU9_1(self.conv9_1(up9))
            conv9 = self.LReLU9_2(self.conv9_2(x))

            latent = self.conv10(conv9)

            if self.opt.times_residual:
                latent = latent * gray

            if self.opt.tanh:
                latent = self.tanh(latent)
            if self.skip:
                if self.opt.linear_add:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
                    input = (input - torch.min(input)) / (torch.max(input) - torch.min(input))
                    output = latent + input * self.opt.skip
                    output = output * 2 - 1
                else:
                    if self.opt.latent_threshold:
                        latent = F.relu(latent)
                    elif self.opt.latent_norm:
                        latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
                    output = latent + input * self.opt.skip
            else:
                output = latent

            if self.opt.linear:
                output = output / torch.max(torch.abs(output))

        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        latent = pad_tensor_back(latent, pad_left, pad_right, pad_top, pad_bottom)
        gray = pad_tensor_back(gray, pad_left, pad_right, pad_top, pad_bottom)
        if flag == 1:
            output = F.upsample(output, scale_factor=2, mode='bilinear')
            gray = F.upsample(gray, scale_factor=2, mode='bilinear')
        if self.skip:
            return output, latent
        else:
            return output
"""
