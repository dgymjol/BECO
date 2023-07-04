# every time
apt-get update
apt-get install sudo
sudo apt-get update
sudo apt-get install unrar
sudo apt-get install libgl1

# only once
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt

pip install matplotlib opencv-python albumentations timm gpustat cython addict chardet
conda install wget git gpustat tensorboard -y
conda install -c conda-forge pydensecrf -y

mkdir data
mkdir data/model_zoo
mkdir data/logging

cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
mv VOCdevkit/VOC2012 .
rm -rf VOCdevkit

# SemgentationAug Images
cd VOC2012
wget --no-check-certificate https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip
unzip SegmentationClassAug.zip
rm -rf __MACOSX
rm SegmentationClassAug.zip

# SegmentationAug List ((DIRECTLY))
# https://github.com/speedinghzl/CCNet/blob/master/dataset/list/voc/train_aug.txt
# location : VOC2012/ImageSets/Segmentation

cd ../model_zoo
wget https://download.openmmlab.com/mmclassification/v0/resnet/resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth
mv resnetv1d101_b32x8_imagenet_20210531-6e13bcd3.pth resnetv1d101_mmcv.pth

wget https://download.pytorch.org/models/resnet101-cd907fc2.pth
mv resnet101-cd907fc2.pth resnet-101_v2.pth

cd ../
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zVCZPhJYiOA3TN3dK4cJhzYHKWPUGdEi' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zVCZPhJYiOA3TN3dK4cJhzYHKWPUGdEi" -O irn_pseudo_label_mask.rar && rm -rf /tmp/cookies.txt
unrar x irn_pseudo_label_mask.rar .