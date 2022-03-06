import os
import random
from data_check import check_format

def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    # files_path = "./MaskDatasets/Annotations"
    files_path = "./CMFD/JPEGImages"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.3

    files_name = sorted([file.split(".")[0] for file in check_format("CMFD")])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        train_f = open("./CMFD/train.txt", "w")
        eval_f = open("./CMFD/valid.txt", "w")
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    main()
