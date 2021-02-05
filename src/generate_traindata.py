import cv2
import pandas as pd
from glob import glob
from random import sample
import os
import pickle
from time import time
import csv

def main():

    root_dir = './'
    # mot_dir = "./MOT20/train/MOT20-01/"
    mot_dir_name = f"{mot_dir.split('/')[-2]}"
    pre_images_path = mot_dir + "img1/"

    thres_visibility = 0.8
    num_per_person = 100
    train_rate = 1#TODO
    ok_num_thres = 6 #TODO: 100から変更
    shuffle = False
    skip_frame = 5
    

    # 出力設定
    output_dir_name_1 = f'../data/{mot_dir_name}_f{num_per_person}_shuffle{shuffle}_skipframe{skip_frame}_oknumthres{ok_num_thres}_3/'
    output_dir_name_2 = f'../data/{mot_dir_name}_f{num_per_person}_shuffle{shuffle}_skipframe{skip_frame}_oknumthres{ok_num_thres}_4/'
    output_1 = root_dir + output_dir_name_1
    output_2 = root_dir + output_dir_name_2
    out_samples_1 = output_1 + 'samples/'
    out_images_1 = output_1 + 'images/'
    out_samples_2 = output_2 + 'samples/'
    out_images_2 = output_2 + 'images/'
    if not os.path.exists(output_1):
        os.mkdir(output_1)
        os.mkdir(output_2)
    else:
        while True:
            print(f"すでに{output_1}は存在しますが続けますか？(y/n) : ", end="")
            inp = input()
            if inp == "y":
                break
            elif inp == "n":
                exit()
            else:
                print("'y' または 'n' をタイプしてください。")
                
    if not os.path.exists(out_samples_1):
        os.mkdir(out_samples_1)
    if not os.path.exists(out_images_1):
        os.mkdir(out_images_1)
    if not os.path.exists(out_samples_2):
        os.mkdir(out_samples_2)
    if not os.path.exists(out_images_2):
        os.mkdir(out_images_2)
    out_train_file_1 = output_1 + 'train.txt'
    out_test_file_1 = output_1 + 'test.txt'
    out_train_file_2 = output_2 + 'train.txt'
    out_test_file_2 = output_2 + 'test.txt'

    gt_file = mot_dir + "gt/gt.txt"

    if os.path.exists(f"./pre_images_{mot_dir_name}.pkl"):
        ts = time()
        pre_images = pickle_load(f"./pre_images_{mot_dir_name}.pkl")
        print(f"読み込み時間: {time() - ts}")
    else:
        pre_images = {int(path.split("/")[-1].split(".")[0]):cv2.imread(path) for path in glob(pre_images_path + "*")}
        ts = time()
        pickle_dump(pre_images, f"./pre_images_{mot_dir_name}.pkl")
        print(f"保存時間: {time() - ts}")

    

    columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "class", "visibility"]
    gt_df = pd.read_csv(gt_file, sep=",", names=columns)

    ids = sorted(list(set(gt_df["id"])))

    ok_num = 0
    for id_ in ids:
        subdf = gt_df[gt_df["id"] == id_]
        if subdf.iloc[0]["conf"] != 1 or subdf.iloc[0]["class"] != 1:
            continue

        subdf = subdf[subdf["visibility"] > thres_visibility]
        # if not(num_per_person*skip_frame/2 <= subdf.shape[0] < num_per_person*skip_frame):
        if subdf.shape[0] < num_per_person*skip_frame:
            continue

        ok_num += 1

        indexs = subdf.index
        if shuffle:
            indexs = sample(list(indexs), num_per_person*skip_frame)
        else:
            indexs1 = list(indexs)[:int(num_per_person*skip_frame/2)]
            indexs2 = list(indexs)[-int(num_per_person*skip_frame/2):]
            indexs = list()
            indexs.extend(indexs1)
            indexs.extend(indexs2)
        sample_save = False
        for k, index_ in enumerate(indexs):
            if k % skip_frame != 0:
                continue
            row = subdf.loc[index_]
            img = pre_images[int(row.frame)]
            x, y, w, h = max(0, int(row.bb_left)), max(0, int(row.bb_top)), int(row.bb_width), int(row.bb_height)
            person_img = img[y:y+h, x:x+w]
            filename = f"F{int(row.frame):05}ID{int(row.id)+100:04}P{-1:02}.png"
            if 2*k < num_per_person*skip_frame:#TODO
                if not sample_save:
                    cv2.imwrite(f"{out_samples_1}/ID{int(row.id)+100}.png", person_img)
                    sample_save = True
                cv2.imwrite(f"{out_images_1}/{filename}", person_img)
                if k < train_rate * num_per_person * skip_frame:
                    with open(out_train_file_1, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{output_dir_name_1}/images/{filename}", int(row.id)+100])
                else:
                    with open(out_test_file_1, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{output_dir_name_1}/images/{filename}", int(row.id)+100])
            else:
                cv2.imwrite(f"{out_images_2}/{filename}", person_img)
                if k < train_rate * num_per_person * skip_frame:
                    with open(out_train_file_2, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{output_dir_name_2}/images/{filename}", int(row.id)+100])
                else:
                    with open(out_test_file_2, "a") as f:
                        writer = csv.writer(f)
                        writer.writerow([f"{output_dir_name_2}/images/{filename}", int(row.id)+100])

        if ok_num == ok_num_thres:
            exit()


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)


    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


if __name__ == "__main__":
    main()