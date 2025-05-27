
https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html

################################ to solve  bad apikey error ################################"
before running the riva init :

ngc download should work in the host :

# install ngc  cli : https://org.ngc.nvidia.com/setup/installers/cli

version 3.50 


modify in the script 3.26=> 3.50

#################################### about model_repository ###################################


it contains models for riva generated using riva init 
it can also contain new modedls destined exclusively for triton server 


riva-start   deploys the models to triton and configures its services (ports , service cascading ...)




##########################################"DOCKER PERMISSION EROR ####################################


sudo usermod -aG docker $USER

newgrp docker
groups $USER

cat /etc/docker/daemon.json
##################################### IF ONLY ruuning riva_start #################################

=>  login docker with key from nvidia 

if riva init +> also ngc ...

https://ngc.nvidia.com/
  +>  setup +> generate personal     key 



#######################" getting large files = models ################################""
sudo apt-get install git-lfs


################################"" tensorrt and cv2   install #############################################"

copy from system 

touti@ubuntu:~/dev/triton_manager$  python -c "import cv2 ; print(cv2.__file__)"
/usr/lib/python3.10/dist-packages/cv2/__init__.py
touti@ubuntu:~/dev/triton_manager$ cp -r  /usr/lib/python3.10/dist-packages/cv2   .venv-tm/lib
lib/   lib64/ 
touti@ubuntu:~/dev/triton_manager$ cp -r  /usr/lib/python3.10/dist-packages/cv2   .venv-tm/lib/python3.10/site-packages/




#########################################  export yolo to tensorrt ########################################

the deployment on triton inference server  works Now :
* the ultra exporter includes metadata in the beginning when writing the engine file 

* modified that behaviour by commeenting  line of code 812-816 ultra/engine/exporter.py 

* idea : save metradata to a file when exporting 

* when calling the model from triton or tensorrt => autobackend is called 

add  

            metadata={

                        "task": "detect",
                        "names":{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'},
                        "stride": 1,
                        "task": "detect",
                        "batch": 1,
                        "imgsz": 640,
            }

 ot laod it from file 

 in line 411  ultra/nn/autobackend.py
