export CUDA_VISIBLE_DEVICES=6

YOUR_OMNIDATA_PATH="/project/aksoy-lab/Mahdi/omnidata/omnidata_tools/torch/"
PRETRAINED_MODELS="/project/aksoy-lab/Mahdi/omnidata/omnidata_tools/torch/pretrained_models/"

# python extract_monocular_cues.py --task depth --img_path ../data/Apartment/scan1/image --output_path ../data/Apartment/scan1 --omnidata_path $YOUR_OMNIDATA_PATH --pretrained_models $PRETRAINED_MODELS
# python extract_monocular_cues.py --task normal --img_path ../data/Apartment/scan1/image --output_path ../data/Apartment/scan1 --omnidata_path $YOUR_OMNIDATA_PATH --pretrained_models $PRETRAINED_MODELS


python extract_monocular_cues.py --task depth --img_path ../data/Studio/scan1/image --output_path ../data/Studio/scan1 --omnidata_path $YOUR_OMNIDATA_PATH --pretrained_models $PRETRAINED_MODELS
python extract_monocular_cues.py --task normal --img_path ../data/Studio/scan1/image --output_path ../data/Studio/scan1 --omnidata_path $YOUR_OMNIDATA_PATH --pretrained_models $PRETRAINED_MODELS
