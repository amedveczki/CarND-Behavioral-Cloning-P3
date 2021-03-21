pip install gdown
mkdir /opt/training
cd /opt/training

gdown --id 1-Vnd721fm-aWzJ63NhHSkJt0saFEGrzk
gdown --id 14S1Oa1NnCm9NefgaTKHfqh3_87oKeiZm
mkdir 1
unzip IMG.zip -d 1
mkdir 2
unzip IMG2.zip -d 2

mkdir IMG
mv 1/IMG/* IMG/
mv -f 2/IMG/* IMG/

cat 1/driving_log.csv 2/driving_log.csv > driving_log.csv
