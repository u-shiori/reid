import cv2
import pandas as pd
from glob import glob
from random import sample
import os
from time import time
import csv

def main(mot_dirname):


    root_dir = "./"
    mot_path = "../data/MOT20/train/"
    assert os.path.isdir(f"{mot_path}/{mot_dirname}"),\
        f"""{mot_path}/{mot_dirname}というディレクトリは存在しません"""
    mot_path += f"/{mot_dirname}"
    pre_images_path = mot_path + "/img1/"

    thres_visibility = 0.8
    num_per_person = 100
    train_rate = 0.8
    shuffle = True
    skip_frame = 1

    # 出力設定
    output_dir_name = f'../data/{mot_dirname}_f{num_per_person}_shuffle{shuffle}_skipframe{skip_frame}/'
    output = root_dir + output_dir_name
    out_images = output + '/images/'

    if not os.path.exists(output):
        os.mkdir(output)

    else:
        while True:
            print(f"すでに{output}は存在しますが続けますか？(y/n) : ", end="")
            inp = input()
            if inp == "y":
                break
            elif inp == "n":
                exit()
            else:
                print("'y' または 'n' をタイプしてください。")

    if not os.path.exists(out_images):
        os.mkdir(out_images)
    out_train_file = output + 'train.txt'
    out_test_file = output + 'test.txt'

    gt_file = mot_path + "/gt/gt.txt"

    
    ts = time()
    pre_images = {int(path.split("/")[-1].split(".")[0]):cv2.imread(path) for path in glob(pre_images_path + "*")}
    print(f"読み込み時間: {time() - ts}")



    columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class", "visibility"]
    gt_df = pd.read_csv(gt_file, sep=",", names=columns)

    ids = sorted(list(set(gt_df["id"])))

    for id_ in ids:
        subdf = gt_df[gt_df["id"] == id_]
        if subdf.iloc[0]["conf"] != 1 or subdf.iloc[0]["class"] != 1:
            continue

        subdf = subdf[subdf["visibility"] > thres_visibility]


        indexs = subdf.index
        if shuffle:
            indexs = sample(list(indexs), num_per_person*skip_frame)
        else:
            indexs1 = list(indexs)[:int(num_per_person*skip_frame/2)]
            indexs2 = list(indexs)[-int(num_per_person*skip_frame/2):]
            indexs = list()
            indexs.extend(indexs1)
            indexs.extend(indexs2)
        for k, index_ in enumerate(indexs):
            if k % skip_frame != 0:
                continue
            row = subdf.loc[index_]
            img = pre_images[int(row.frame)]
            x, y, w, h = max(0, int(row.bb_left)), max(0, int(row.bb_top)), int(row.bb_width), int(row.bb_height)
            person_img = img[y:y+h, x:x+w]
            filename = f"F{int(row.frame):05}ID{int(row.id)+100:04}P{-1:02}.png"
            if k < num_per_person*skip_frame:
                cv2.imwrite(f"{out_images}/{filename}", person_img)
                if k < train_rate * num_per_person * skip_frame:
                    with open(out_train_file, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{output_dir_name}/images/{filename}", int(row.id)])
                else:
                    with open(out_test_file, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{output_dir_name}/images/{filename}", int(row.id)])






if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument(
        '--mot_dirname',
        default="MOT20-03",
        type=str,
        help='MOT20のうちのどのディレクトリを使用するか',
    )
    
    args   = parser.parse_args()
    main(args.mot_dirname)