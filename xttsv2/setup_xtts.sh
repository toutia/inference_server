#!/bin/bash
set -e

# Output directory
MODEL_DIR="xtts/1/xtts_model"



# Download XTTSv2 model
echo "Downloading XTTSv2 model files from Hugging Face..."
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth -O "$MODEL_DIR/model.pth"
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json -O "$MODEL_DIR/config.json"
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/speakers_xtts.pth -O "$MODEL_DIR/speakers_xtts.pth"
wget https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json -O "$MODEL_DIR/vocab.json"


sudo chown -R touti:touti ../model_repository

cp -rf   xtts  ../model_repository/models


echo "XTTSv2 model downloaded to $MODEL_DIR and copied to $MODEL_REPO "


