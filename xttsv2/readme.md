######################### deploy xttsv2 to triton ##########################
cd xttsv2 

./setup_xtts.sh

python   speech_synthesizer.py 


# create a ros2 node : 
get text from asr taranscript topic created by recognizer 
sythesises using xttsv2 see tts_french.py 
for using the model directly see 

models/
└── xttsv2/
    ├── config.pbtxt
    └── 1/
        ├── model.py
        ├── xtts_model/       # your model files

        found hee : /home/touti/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v

then deploy the model into triton (riva container) will be changed using docker file to install TTS python module

for torch and corresponding torch audio see above 
                        XTTS on Jetson AGX Orin - Summary of Issues
                        getting torch version for cuda 12.6
                        https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/



                        # when installing torch or torchaudio (should have the same version) 
                        nvcc --version  = cuda version 

                        Your CUDA Version	Recommended PyTorch Version	Installation Method
                        CUDA 12.2+	PyTorch 2.4 / 2.5 (build from src)	Manual source build
                        CUDA 11.8	PyTorch 2.0–2.5	Official pip/conda install
                        CUDA 11.6	PyTorch ≤2.2	Pip with cu116 wheel
                        CUDA <11.0	PyTorch ≤1.10	Older pip wheels or source build required



                        # when building  torchaudio using 


                        for torch 

                        git clone --recursive --branch v2.5.0 https://github.com/pytorch/pytorch.git
                        cd pytorch

                        # (Optional) Set up conda env with dependencies
                        pip install -r requirements.txt

                        # Build and install  
                        python setup.py install
                        # to get the wheel just 
                        python setup.py bdist_wheel




                        git clone --branch release/2.5 https://github.com/pytorch/audio.git
                        cd audio
                        python setup.py install



                        # build locally and copy through docker file semes to be the best option 

                        COPY ./wheels/*.whl /wheels/
                        RUN pip install /wheels/*.whl

                        FROM nvcr.io/nvidia/riva/riva-speech:2.16.0-l4t-aarch64

                        # TTS can have git based packages ? 
                        RUN apt-get update && apt-get install -y git

                        # Install Coqui TTS and its dependencies
                        RUN pip install TTS==0.22.0
                        RUN pip install /wheels/*.whl





                        1. Incompatible PyTorch or LibTorch binary: Attempting to link against x86_64 binaries on aarch64
                        Jetson AGX Orin caused linker errors (wrong format).
                        2. Torchaudio build failure: Compilation errors due to missing symbols and CUDA-related issues,
                        including undefined symbols from libtorch and improper linking.


                        =   sudo rm /usr/local/lib/libtorch*.so

                        sudo rm /usr/local/lib/libc10*.so
                        due to imropoer installation of torch 


                        3. NumPy version mismatch: Runtime warning triggered due to incompatible compiled and runtime
                        NumPy versions.
                        pip install numpy==1.26.1

                        4. cudnn missing: Torch failed to import due to missing libcudnn.so.8, which is essential for GPU
                        acceleration.
                        install the right version of torchaudio corresponding torch version 2.5 



                        5. Compilation errors due to missing definitions: `FLT_MAX` not defined in CUDA kernel. Solution:
                        `#include <cfloat>` was missing. 
                        in the audio folder get the file path from error 



