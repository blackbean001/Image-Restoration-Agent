# set pytorch:2.2.2-cuda12.1-cudnn8-deval as the base
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Copy application files
#COPY ./AgenticIR ./
#COPY ./CLIP4Cir ./

# set environment
ENV DEBIAN_FRONTEND=noninteractive

# build postgresql environment
RUN apt-get update  \
  && apt-get install -y git vim unzip curl  \
  && apt-get install -y postgresql postgresql-client  \
  && apt-get install git-lfs && git-lfs install  \
  && pip install pgvector  && apt install -y postgresql-server-dev-14  \
  && git clone --branch v0.8.1 https://github.com/pgvector/pgvector.git && cd pgvector && make && make install && cd .. \
  && apt-get install -y libpq-dev && pip install psycopg2  \
  && sed -i "$(($(wc -l < /etc/postgresql/14/main/pg_hba.conf)-20+1)),\$ s/\bpeer\b/trust/" /etc/postgresql/14/main/pg_hba.conf  \
  && /etc/init.d/postgresql start \
  && psql -U postgres -c "CREATE DATABASE agenticir_rag_test;" \ 
  && apt-get install -y systemd

# download model weights
RUN pip install gdown==5.2.0  \
  && cd /app/./AgentApp/weights/defocus_deblurring/DRBNet && python download_ckpts.py  \
  && cd /app/./AgentApp/weights/defocus_deblurring/IFAN && sh download_ckpts.sh  \
  && cd /app/./AgentApp/weights/defocus_deblurring/Restormer && python download_ckpt.py  \
  && cd /app/./AgentApp/weights/dehazing/DehazeFormer && python download_ckpt.py  \
  && cd /app/./AgentApp/weights/dehazing/RIDCP_dehazing && python download_ckpts.py  \
  && cd /app/./AgentApp/weights/dehazing/X-Restormer && python download_ckpts.py  \
  && cd /app/./AgentApp/weights/dehazing/maxim && python download_ckpts.py \
  && cd /app/./AgentApp/weights/denoising/MPRNet && python download_ckpts.py \
  && cd /app/./AgentApp/weights/denoising/SwinIR && python download_ckpts.py \
  && cd /app/./AgentApp/weights/jpeg_compression_artifact_removal/FBCNN && sh download_ckpt.sh \
  && cd /app/./AgentApp/weights/super_resolution/DiffBIR && sh download_ckpt.sh \
  && cd /app/./AgentApp/weights/super_resolution/HAT && python download_ckpts.py \
  && cd /app/./AgenticIR/retrival_database/CLIP4CIR/models && python download_ckpts.py \
\
  && cd /app/./AgenticIR/DepictQA/weights && sh download_ViT-L-14.sh  \
  && sh download_vicuna-7b-v1.5.sh && cd delta && sh download_Abstractor.sh && sh download_degra_eval.sh  \

# link weights in AgenticIR to AgenticApp
RUN ln -s /app/./AgentApp/weights/defocus_deblurring/DRBNet/ckpts/* /app/./AgenticIR/executor/defocus_deblurring/tools/DRBNet/ckpts  \
  && mkdir -p /app/./AgenticIR/executor/defocus_deblurring/tools/IFAN/ckpt && ln -s /app/./AgentApp/weights/defocus_deblurring/IFAN/* /app/./AgenticIR/executor/defocus_deblurring/tools/IFAN/ckpt/  \
  && ln -s /app/./AgentApp/weights/defocus_deblurring/Restormer/Defocus_Deblurring/* /app/./AgenticIR/executor/defocus_deblurring/tools/Restormer/Defocus_Deblurring/pretrained_models  && ln -s /app/./AgentApp/weights/defocus_deblurring/Restormer/Denoising/* /app/./AgenticIR/executor/defocus_deblurring/tools/Restormer/Denoising/pretrained_models && ln -s /app/./AgentApp/weights/defocus_deblurring/Restormer/Deraining/* /app/./AgenticIR/executor/defocus_deblurring/tools/Restormer/Deraining/pretrained_models && ln -s /app/./AgentApp/weights/motion_deblurring/Restormer/Motion_Deblurring/* /app/./AgenticIR/executor/motion_deblurring/tools/Restormer/Motion_Deblurring/pretrained_models  \
  && ln -s /app/./AgentApp/weights/dehazing/DehazeFormer/saved_models /app/./AgenticIR/executor/dehazing/tools/DehazeFormer && ln -s /app/./AgentApp/weights/dehazing/DehazeFormer/saved_models/indoor/dehazeformer-b.pth /app/./AgentApp/weights/dehazing/DehazeFormer/saved_models/  \
  && mkdir -p /app/./AgenticIR/executor/dehazing/tools/RIDCP_dehazing/pretrained_models && ln -s /app/./AgentApp/weights/dehazing/RIDCP_dehazing/pretrained_RIDCP.pth /app/./AgenticIR/executor/dehazing/tools/RIDCP_dehazing/pretrained_models  \
  && mkdir -p /app/./AgenticIR/executor/dehazing/tools/X-Restormer/experiments/pretrained && ln -s /app/./AgentApp/weights/dehazing/X-Restormer/*.pth /app/./AgenticIR/executor/dehazing/tools/X-Restormer/experiments/pretrained  \
  && mkdir -p /app/./AgenticIR/executor/dehazing/tools/maxim/maxim/checkpoints && ln -s /app/./AgentApp/weights/dehazing/maxim/* /app/./AgenticIR/executor/dehazing/tools/maxim/maxim/checkpoints  \
  && ln -s /app/./AgentApp/weights/motion_deblurring/MPRNet/model_deblurring.pth /app/./AgenticIR/executor/denoising/tools/MPRNet/Deblurring/pretrained_models/ && ln -s /app/./AgentApp/weights/motion_deblurring/MPRNet/model_denoising.pth /app/./AgenticIR/executor/denoising/tools/MPRNet/Denoising/pretrained_models/ && ln -s /app/./AgentApp/weights/motion_deblurring/MPRNet/model_deraining.pth /app/./AgenticIR/executor/denoising/tools/MPRNet/Deraining/pretrained_models/  \
  && ln -s /app/./AgentApp/weights/jpeg_compression_artifact_removal/FBCNN/*.pth /app/./AgenticIR/executor/jpeg_compression_artifact_removal/tools/FBCNN/model_zoo  \
  && ln -s /app/./AgentApp/weights/jpeg_compression_artifact_removal/SwinIR/*.pth /app/./AgenticIR/executor/jpeg_compression_artifact_removal/tools/SwinIR/model_zoo  \
  && mkdir -p /app/./AgenticIR/executor/super_resolution/tools/DiffBIR/weights && ln -s /app/./AgentApp/weights/super_resolution/DiffBIR/* /app/./AgenticIR/executor/super_resolution/tools/DiffBIR/weights  \
  && ln -s /app/./AgentApp/weights/super_resolution/HAT/*.pth /app/./AgenticIR/executor/super_resolution/tools/HAT/experiments/pretrained_models

# build environment
RUN pip install numpy==1.24.1 torch==2.1.0 opencv-python==4.8.0.76 \
  && rm /bin/sh && ln -s /bin/bash /bin/sh  \
\
  && conda create -y -n clip4cir -y python=3.8  \
  && source activate clip4cir  \
  && conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0  \
  && conda install -y -c anaconda pandas=1.4.2 \
  && pip install comet-ml==3.21.0, urllib3==1.26.18  \
  && pip install git+https://github.com/openai/CLIP.git  \
  && pip install pgvector  \
  && apt-get install -y libpq-dev && pip install psycopg2  \
\
  && conda create -y -n agenticir python=3.10  \
  && source activate agenticir  \
  && apt-get install -y ffmpeg libsm6 libxext6 curl  \
  && cd /app/AgenticIR && pip install -r installation/requirements.txt  \
  && pip install git+https://github.com/openai/CLIP.git \
  && pip install langgraph fastapi python-multipart uvicorn \
\
  && conda create -y -n depictqa python=3.10  \
  && source activate depictqa  \
  && pip install -r /app/AgenticIR/DepictQA/requirements.txt  \
  && pip install flask huggingface-hub==0.23.0  \
  && cd /app/AgenticIR/DepictQA && sh launch_service.sh  \
\
  && conda create -y -n dehazeformer python=3.10  \
  && source activate dehazeformer  \
  && pip install flask flask_cors ptflops lpips pyyaml  \
  && pip install -r /app/AgenticIR/executor/dehazing/tools/DehazeFormer/requirements.txt  \
\
  && conda create -y -n diffbir python=3.10  \
  && source activate diffbir \
  && pip install flask flask_cors ptflops lpips pyyaml  \
  && pip install -r /app/AgenticIR/executor/super_resolution/tools/DiffBIR/requirements.txt  \
\
  && conda create -y -n drbnet python=3.8  \
  && source activate drbnet  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && apt-get install jq \
  && pip install -r /app/AgenticIR/executor/defocus_deblurring/tools/DRBNet/requirements.txt  \
\
  && conda create -y -n fbcnn python=3.10.18  \
  && source activate fbcnn  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && pip install -r  /app/AgenticIR/executor/jpeg_compression_artifact_removal/tools/FBCNN/requirements.txt \
\
  && conda create -y -n hat python=3.10.18  \
  && source activate hat  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && pip install -r /app/AgenticIR/executor/super_resolution/tools/HAT/requirements.txt  \
  && python setup.py develop  \
\
  && conda create -y -n ifan python=3.8.20  \
  && source activate ifan  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && pip install -r /app/AgenticIR/executor/defocus_deblurring/tools/IFAN/requirements.txt  \
\
  && conda create -y -n maxim python=3.10.18  \
  && source activate maxim  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && pip install -r /app/AgenticIR/executor/denoising/tools/maxim/requirements.txt  \
  && pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html  \
  && pip install .  \
\
  && conda create -y -n mprnet python=3.7.16  \
  && source activate mprnet  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && conda install -y pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch  \
  && pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm  \
  && pip install "Pillow<7"  \
  && cd /app/./AgenticIR/executor/denoising/tools/MPRNet/pytorch-gradual-warmup-lr && python setup.py install  \
\
  && conda create -y -n restormer python=3.7.16  \
  && source activate restormer  \
  && pip install -r /app/AgenticIR/executor/denoising/tools/Restormer/requirements.txt  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && pip install numpy==1.26.4  \
\ 
  && conda create -y -n ridcp python=3.8.20  \
  && source activate ridcp  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && pip install -r /app/AgenticIR/executor/dehazing/tools/RIDCP_dehazing/requirements.txt  \
\
  && conda create -y -n swinir python=3.10.18  \
  && source activate swinir  \
  && pip install flask flask_cors ptflops lpips pyyaml requests timm \
  && pip install -r /app/AgenticIR/executor/denoising/tools/SwinIR/requirements.txt  \
\
  && conda create -y -n xrestormer python=3.10.18  \
  && source activate xrestormer  \
  && pip install flask flask_cors ptflops lpips pyyaml \
  && pip install -r /app/AgenticIR/executor/denoising/tools/X-Restormer/requirements.txt  \
  && pip install numpy==1.26.4  \
  && python setup.py develop  \
  && sed -i '8s/.*/from torchvision.transforms.functional import rgb_to_grayscale/' /opt/conda/envs/xrestormer/lib/python3.10/site-packages/basicsr/data/degradations.py

# test conda environment
RUN if ["$do_test" == "true"]; then cd /app/AgenticIR && sh test_env.sh; fi


