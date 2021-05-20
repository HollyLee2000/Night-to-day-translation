import os
from PIL import Image

"""
此文件的作用是制造配对训练集，方便利用轻巧的网络结构，如datasets/alige下有trainA、trainB、testA、testB四个文件夹，那么使用
python datasets/make_dataset_aligned.py --dataset-path datasets/alige 可以制造配对训练图像于testA、testB
"""


def get_file_paths(folder):
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                image_file_paths.append(file_path)
        break  # 防止下降到子文件夹中
    return image_file_paths


def align_images(a_file_paths, b_file_paths, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for i in range(len(a_file_paths)):
        img_a = Image.open(a_file_paths[i])
        img_b = Image.open(b_file_paths[i])
        print(str(img_a) + " " + str(img_b))
        assert (img_a.size == img_b.size)

        aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))
        aligned_image.save(os.path.join(target_path, '{:04d}.jpg'.format(i)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        dest='dataset_path',
    )
    args = parser.parse_args()

    dataset_folder = args.dataset_path
    test_a_path = os.path.join(dataset_folder, 'testA')
    test_b_path = os.path.join(dataset_folder, 'testB')
    test_a_file_paths = get_file_paths(test_a_path)
    test_b_file_paths = get_file_paths(test_b_path)
    assert (len(test_a_file_paths) == len(test_b_file_paths))
    test_path = os.path.join(dataset_folder, 'test')

    train_a_path = os.path.join(dataset_folder, 'trainA')
    train_b_path = os.path.join(dataset_folder, 'trainB')
    train_a_file_paths = get_file_paths(train_a_path)
    train_b_file_paths = get_file_paths(train_b_path)
    assert (len(train_a_file_paths) == len(train_b_file_paths))
    train_path = os.path.join(dataset_folder, 'train')

    align_images(test_a_file_paths, test_b_file_paths, test_path)
    align_images(train_a_file_paths, train_b_file_paths, train_path)
