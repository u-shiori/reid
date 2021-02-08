mkdir data
cd data
wget https://motchallenge.net/data/MOT20.zip
unzip MOT20.zip

cd src
python generate_traindata.py --mot_dirname MOT20-03
cd ..