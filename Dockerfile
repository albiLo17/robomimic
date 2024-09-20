FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    git python3-dev python3-pip \
    libxrender1 \
    libxxf86vm-dev \
    libxfixes-dev \
    libxi-dev \
    libxkbcommon-dev \
    libsm-dev \
    libgl-dev \
    python3-tk \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

RUN pip install torch \
torchvision \
torchaudio \
matplotlib \
argparse \
lpips \
plyfile \
imageio-ffmpeg \
h5py \
imageio \
natsort \
numpy \
wandb \
seaborn 

# ################ old
# # # Use NVIDIA's CUDA-enabled base image with Python 3.8 and CUDA 11.7
# # FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# # Install dependencies and Miniconda
# RUN apt-get update && apt-get install -y \
#     git \
#     wget \
#     && rm -rf /var/lib/apt/lists/*

# # Install Miniconda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
#     rm Miniconda3-latest-Linux-x86_64.sh

# # Set the working directory in the container
# WORKDIR /usr/workspace

# # Set environment variables to automatically configure timezone
# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y tzdata


# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender-dev

# RUN apt-get update && apt-get install -y \
#     libglvnd-dev \
#     libglew-dev \
#     libgl1-mesa-glx \
#     libosmesa6 \
#     libgl1-mesa-dev \
#     mesa-utils \
#     xvfb




# # #RUN apt-get update && apt-get install -y cuda

# # Set up the environment
# ENV PATH /opt/conda/bin:$PATH

# # Create and activate a conda environment with Python 3.8
# RUN conda create -n robomimic_venv python=3.8.0 && \
#     echo "source activate robomimic_venv" > ~/.bashrc

# # Set the environment to use the conda environment by default
# ENV PATH /opt/conda/envs/robomimic_venv/bin:$PATH
# ENV CONDA_DEFAULT_ENV robomimic_venv

# ########### START FROM WHERE WE LEFT OFF ###########

# # Install PyTorch with CUDA support
# RUN pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# RUN apt-get update && apt-get install -y git

# # Install additional system libraries required for egl_probe
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libegl1-mesa \
#     libgles2-mesa \
#     libglfw3 \
#     libxrandr2 \
#     libxinerama1 \
#     libxcursor1 \
#     libxi6 \
#     mesa-utils \
#     freeglut3-dev

# # Install Python build tools and system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     python3-dev \
#     cmake \
#     libffi-dev \
#     libssl-dev \
#     libglu1-mesa-dev \
#     libglew-dev


# # Clone egl_probe repository and install from source
# # RUN cd  /usr/workspace && \
# #     git clone git@github.com:StanfordVL/egl_probe.git&& \
# #     cd egl_probe && \
# #     python setup.py install

# RUN pip install egl_probe

# # Clone robomimic from source and install in editable mode
# RUN cd  /usr/workspace && \
#     git clone https://github.com/albiLo17/robomimic.git && \
#     cd robomimic && \
#     pip install -e .

# # Install robosuite from source
# RUN cd /usr/workspace && \
#     git clone https://github.com/ARISE-Initiative/robosuite.git && \
#     cd robosuite && \
#     pip install -r requirements.txt

# # Optionally switch to v1.4.1 branch for robosuite
# RUN cd /usr/workspace/robosuite && \
#     git checkout v1.4.1

# # Install NVIDIA's CUDA libraries
# RUN conda install -y -c conda-forge cudnn==8.1.0.77

# # Ensure that CUDA is visible inside the container
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# # install this somewhere
# RUN cd /usr/workspace && \
#     git clone https://github.com/quasimetric-learning/torch-quasimetric.git && \
#     cd torch-quasimetric && \
#     python setup.py install && \
#     pip install wandb


# # Set environment variable for OSMesa rendering in MuJoCo
# ENV MUJOCO_GL=osmesa

# # Set the default command to open a bash terminal in the robomimic environment
# CMD [ "bash" ]
