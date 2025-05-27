FROM nvcr.io/nvidia/riva/riva-speech:2.16.0-l4t-aarch64

# TTS can have git based packages ? 
RUN apt-get update && apt-get install -y git
RUN apt install build-essential ninja-build -y
# rus is needed by TTS
RUN apt-get install -y curl build-essential

# Install Rust using rustup (preferred way)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"



# # Install Coqui TTS and its dependencies
# # to fix the build cause the container has 59.6 ..
RUN pip install --upgrade pip setuptools wheel build
# RUN pip install  --no-cache-dir --ignore-installed TTS==0.22.0
RUN pip install --no-cache-dir --ignore-installed  git+https://github.com/toutia/TTS.git


# the container version contains cuda 12.2  = torch 2.2 / 24 01  #https://developer.nvidia.com/embedded/downloads
# so install  torch 2.3 and torchaudio same version 
# wheels for aarch64 not available building directly here 


RUN pip install cmake==3.30.5


# RUN git clone --recursive --branch v2.3.0 https://github.com/pytorch/pytorch.git
# WORKDIR /opt/riva/pytorch
# RUN pip install -r requirements.txt

# # Build and install  
ENV TORCH_CUDA_ARCH_LIST="8.7;8.9"
# RUN python3 setup.py install
# WORKDIR /opt/riva

# RUN pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu121


#https://developer.nvidia.com/embedded/downloads   2.2 2401 is retained
COPY ./wheels/*.whl /wheels/
RUN pip install /wheels/*.whl

# RUN pip uninstall torchaudio -y

RUN git clone --branch release/2.2 https://github.com/pytorch/audio.git
WORKDIR /opt/riva/audio
RUN python3 setup.py install


# attempting to fix 'GPT2InferenceModel' object has no attribute 'generate' issue starting from 4.50.0  =   WORKS

RUN pip install transformers==4.49.0







