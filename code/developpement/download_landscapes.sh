cd data
wget https://github.com/ml5js/ml5-data-and-models/raw/master/datasets/images/landscapes/landscapes_small.zip
mkdir landscapes
unzip landscapes_small.zip -d landscapes
rm landscapes_small.zip
rm -r landscapes/__MACOSX
cd ..