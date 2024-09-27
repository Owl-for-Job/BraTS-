import h5py
import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def generate_txt_file(data_root, output_file):
    # 获取数据根目录下的所有文件夹名称
    patients = [name for name in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]

    # 过滤掉不符合命名规范的文件夹
    patients = [name for name in patients if name.startswith('BraTS2021_')]

    # 提取患者ID
    patient_ids = [name.split('_')[-1] for name in patients]

    # 将患者ID写入txt文件
    with open(output_file, 'w') as f:
        for patient_id in patient_ids:
            f.write(f"{patient_id}\n")

    print(f"Generated {output_file} with {len(patient_ids)} patient IDs.")


# 设置数据集根目录和输出文件路径
data_root = "E:\\DataSet\\Data\\BraTS2021_Training_Data"
output_file = os.path.join(data_root, 'train.txt')

# 生成txt文件，仅在第一次运行时需要，后续可以注释掉
generate_txt_file(data_root, output_file)




# 四种模态的mri图像
modalities = ['flair', 't1ce', 't1', 't2']

# train
train_set = {
        'root':"E:\\DataSet\\Data\\BraTS2021_Training_Data",  # 四个模态数据所在地址
        'out': "E:\\DataSet\\Data\\output_temp",  # 预处理输出地址
        'flist': 'train.txt',  # 训练集名单（有标签）
        }

import nibabel as nib
def process_h5(path, out_path):
    """ Save the data with dtype=float32.
        z-score is used but keep the background with zero! """
    try:
        label = nib.load(os.path.join(path+"seg.nii.gz")).get_fdata()

        X=[]
        # 读取并堆叠NIFTI图像
        for j in modalities:
            img_path = os.path.join(path+f"{j}.nii.gz")
            img = nib.load(img_path).get_fdata()
            #img = (img - data_types_mean_std_dict[j]['mean']) / data_types_mean_std_dict[j]['std']
            img = img.astype(np.float32)
            X.append(img)
        images = np.stack(X, 0)
        # 打印堆叠后的图像形状
        print(f"Stacked image shape: {images.shape}")
        #（240,240,155,4）

        # 数据类型转换
        label = label.astype(np.uint8)
        images = images.astype(np.float32)
        case_name = os.path.basename(path)

        path = os.path.join(out_path, case_name)
        print("path=",path)
        output = path + 'mri_norm2.h5'
        # 对第一个通道求和，如果四个模态都为0，则标记为背景(False)
        mask = images.sum(0) > 0
        for k in range(4):
            x = images[k, ...]  #
            y = x[mask]

            # 对背景外的区域进行归一化
            x[mask] -= y.mean()
            x[mask] /= y.std()

            images[k, ...] = x
        print(case_name, images.shape,label.shape)#, label.shape
        f = h5py.File(output, 'w')
        f.create_dataset('image', data=images, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

    except Exception as e:
        # 如果遇到错误，打印警告并跳过该文件
        print(f"Error: {e}\n")



def doit(dset):
    root, out_path = dset['root'], dset['out']
    file_list = os.path.join(root, dset['flist'])
    subjects = open(file_list).read().splitlines()
    names = ['BraTS2021_' + sub for sub in subjects]
    paths = [os.path.join(root, name, name + '_') for name in names]

    for path in tqdm(paths):
        print(path)
        process_h5(path, out_path)
        # break
    print('Finished')


if __name__ == '__main__':
    #主函数
    doit(train_set)
    import h5py
    import numpy as np
    #打印00000号样本用于检查是否格式正确
    p = "E:\DataSet\Data\output\BraTS2021_00000_mri_norm2.h5"
    h5f = h5py.File(p, 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    print('image shape:', image.shape, '\t', 'label shape', label.shape)
    print('label set:', np.unique(label))

    # image shape: (4, 240, 240, 155)          label shape (240, 240, 155)
    # label set: [0 1 2 4]


