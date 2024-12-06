FROM python:3.11

# Set the working directory inside the container
WORKDIR /code

# Install system dependencies for OpenCV and others
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libffi-dev \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Copy the requirements file and install dependencies
COPY requirements-locked.txt /code/requirements-locked.txt
RUN pip install -r /code/requirements-locked.txt

# Copy the project files to the container
COPY ByteTrack /code/ByteTrack
COPY bytetrackCustom /code/bytetrackCustom
COPY config /code/config
COPY helpers /code/helpers
COPY lines_data/cam_line_data_3_3_2.csv /code/lines_data
COPY pretrained_model_wgt/infer_script.py /code

# Set the default command to run the infer script
CMD ["python", "infer_script.py"]

