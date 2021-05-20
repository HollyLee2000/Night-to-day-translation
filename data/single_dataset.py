from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    """
    此数据集类可以加载由path --dataroot /path/to/data指定的一组图像。
    用于在cycle-GAN模型测试中只产生一方映射的结果
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """
            A(tensor) - - 域中的一个图像
            A_paths(str) - - 图像的路径
        """
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)
