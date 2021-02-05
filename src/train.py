import os

from time import time
import optuna
import logging
import datetime

from datasets.getDataLoader import getDataLoader
from models.getModel import getModel
from losses.getLoss import getLoss
from trainers.getTrainer import getTrainer

from _utils import getLogger, setLogLevel, setLogFile
pil_logger = getLogger('PIL')
pil_logger.setLevel(logging.INFO)
logger = getLogger(__name__)


#"Triplet" or "Arcface" or "ImprovedTriplet" or "Quadruplet" or "UIR"
model_name = "ClusterTriplet"

model_names = ["Triplet", "Arcface", "ImprovedTriplet", "Quadruplet", "ClusterTriplet", "UIR"]
assert model_name in model_names,\
    f"""model {model_name} is not supported. Choose model in {model_names}."""


# optuna parameter
OPT = False
if OPT:
    n_trials = 100
    timeout = 86400
    num_out = None
else:
    if model_name=="Triplet" or model_name=="ImprovedTriplet" or model_name=="Quadruplet" or model_name=="ClusterTriplet":
        num_out = 206
    elif model_name == "Arcface" or model_name=="UIR":
        num_out = 645

# train parameter
n_epochs = 30
log_interval = 40
save_epoch_interval = 1
margin_a = 1. 
margin_b = 1. if model_name == "ImprovedTriplet" or model_name=="Quadruplet" or model_name=="ClusterTriplet" else None
w = 0.1 if model_name == "UIR" else None
cluster_distance_name = "Single" if model_name == "ClusterTriplet" else None
if model_name=="ClusterTriplet":
    cluster_distance_names = ["Single","Complete","Centroid","Average","Ward"]
    assert cluster_distance_name in cluster_distance_names,\
        f"""cluster distance {model_name} is not supported. Choose distance in {cluster_distance_names}."""
in_channels = 3
lr = 1e-3

start_epoch = 0

batch_size = 32#128
thread = 0


# config data path
cfg_train_path = "../data/MOT20-03/train.txt"
cfg_valid_path = "../data/MOT20-03/test.txt"
cfg_test_path = "../data/MOT20-03/test.txt"

"""Setting end."""




if model_name == "Arcface":
    outdir = "../result/checkpoints/arcface/"
elif model_name == "Triplet":
    outdir = "../result/checkpoints/triplet/"
elif model_name == "ImprovedTriplet":
    outdir = "../result/checkpoints/improved_triplet/"
elif model_name == "Quadruplet":
    outdir = "../result/checkpoints/quadruplet/"
elif model_name == "ClusterTriplet":
    outdir = f"../result/checkpoints/cluster_triplet/{cluster_distance_name}"
elif model_name == "UIR":
    outdir = "../result/checkpoints/uir/"
else:
    logging.error(f"outdir has not been set.")
    exit(1)


if not os.path.exists(outdir):
    if start_epoch == 0:
        os.makedirs(outdir)
else:
    while True:
        print(f"すでに{outdir}は存在しますが続けますか？(y/n) : ", end="")
        inp = input()
        if inp == "y":
            break
        elif inp == "n":
            exit()
        else:
            print("'y' または 'n' をタイプしてください。")

setLogFile(outdir+f'/{datetime.datetime.now()}.log')
setLogLevel(10)

logging.info("-"*30)
logging.info("--- パラメータ ---\n")
logging.info(f"model: {model_name}")
logging.info(f"cluster distance: {cluster_distance_name}")
logging.info(f"n epochs : {n_epochs}")
logging.info(f"train data: {cfg_train_path}")
logging.info(f"valid data: {cfg_valid_path}")
logging.info(f"test data: {cfg_test_path}")
logging.info("\n")

data_dirname = cfg_train_path.split("/")[-2]



#Dataloaderの取得
sup_train_loader, semisup_train_loader, class_num \
    = getDataLoader(model_name, cfg_train_path, batch_size, train=True, thread=thread)
sup_test_loader, semisup_test_loader, test_class_num \
    = getDataLoader(model_name, cfg_test_path, batch_size, train=False, thread=thread)
sup_valid_loader, semisup_valid_loader, valid_class_num \
    = getDataLoader(model_name, cfg_valid_path, batch_size, train=False, thread=thread)

assert class_num == test_class_num,\
    f"class_numが,trainとtestで異なります．揃えてください．\n\ttrain={class_num},test={test_class_num}"
logging.info(f"class_num(train,test): {class_num}")
logging.info(f"class_num(valid): {valid_class_num}")
logging.info("\n")






def objective(trial):
    num_out = trial.suggest_int('num_out', 2, 1024)
    logging.info(f"num_out = {num_out}")

    #Modelの取得
    model = getModel(model_name, num_out)
    #Lossの取得
    margin_penalty, sup_train_loss_fn, semisup_train_loss_fn, test_loss_fn = getLoss(model_name, num_out, class_num, margin_a, margin_b, w)
    #Trainerの取得
    trainer = getTrainer(model_name, sup_train_loader, semisup_train_loader, \
        sup_valid_loader, semisup_valid_loader, sup_test_loader, semisup_test_loader, \
        model, margin_penalty, sup_train_loss_fn, semisup_train_loss_fn, test_loss_fn)
    #学習
    score = trainer.fit(lr, n_epochs, log_interval, save_epoch_interval, start_epoch=0, outdir=outdir, data_dirname=data_dirname)

    return score



#optunaの探索
if OPT:
    st = time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    logging.info(f"all time = {time() - st}")
    logging.info("\n" + "-"*30)
    logging.info('Best trial:')
    trial = study.best_trial
    logging.info(f'  Value: {trial.value}')
    logging.info('  Params: ')
    for key, value in trial.params.items():
        logging.info(f'    {key}: {value}')  
    logging.info("-"*30 + "\n")
    
    num_out = trial.params["num_out"]

#本実行

#Modelの取得
model = getModel(model_name, num_out)
#Lossの取得
margin_penalty, sup_train_loss_fn, semisup_train_loss_fn, test_loss_fn = getLoss(model_name, num_out, class_num, margin_a, margin_b, w)
#Trainerの取得
trainer = getTrainer(model_name, sup_train_loader, semisup_train_loader, sup_valid_loader, semisup_valid_loader, sup_test_loader, semisup_test_loader, \
                model, margin_penalty, sup_train_loss_fn, semisup_train_loss_fn, test_loss_fn, cluster_distance_name)

logging.info(f"model: {model}")
logging.info(f"margin_penalty: {margin_penalty}")
logging.info(f"train_loss_fn: {sup_train_loss_fn}")
logging.info(f"train_loss_fn: {semisup_train_loss_fn}")
logging.info(f"test_loss_fn: {test_loss_fn}")
logging.info(f"trainer: {trainer}")
logging.info("-"*30)

#学習
score = trainer.fit(lr, n_epochs, log_interval, save_epoch_interval, start_epoch=start_epoch, outdir=outdir, data_dirname=data_dirname)
