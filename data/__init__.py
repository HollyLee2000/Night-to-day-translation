import importlib
import torch.utils.data
from data.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """
    此模块的作用是将名为[dataset_name]_dataset.py的类实例化。它必须是BaseDataset的一个子类，不区分大小写
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)  # 动态导入对象
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():  # 遍历py文件中的class，如果转为小写后和所找文件名一致并且为BaseDataset子类，则返回
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """返回数据集类的静态方法"""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """
    创建一个给定该选项的数据集。
    此函数包装CustomDatasetDataLoader类,这是data包与“train.py/test.py”之间的主要接口
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    """执行多线程数据加载的数据集类的包装类 """
    def __init__(self, opt):
        """
        创建一个名为[dataset_mode]的实例,再创建一个多线程的数据加载
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,  # 批处理大小
            shuffle=not opt.serial_batches,  # 设为true表示数据每一代会重组
            num_workers=int(opt.num_threads))  # 线程数，windows系统一般只能设0

    def load_data(self):
        return self

    def __len__(self):
        """返回数据集中的数据数量 """
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """返回一批的数据"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data  # yield关键字表示返回后从此处继续运行
