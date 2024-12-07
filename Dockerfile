# Start with a base image
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

# Copy only the requirements file to leverage caching
COPY requirements-locked.txt /code/requirements-locked.txt

# Install Python dependencies (this step will be cached until the requirements file changes)
RUN pip install --no-cache-dir -r /code/requirements-locked.txt

# Copy the project files to the container (only necessary code)
COPY ByteTrack /code/ByteTrack
COPY bytetrackCustom /code/bytetrackCustom
COPY config /code/config
COPY helpers /code/helpers

# Copy the specific data file for caching purposes
COPY lines_data/cam_line_data_3_3_2.csv /code/lines_data/

# Copy any other files that do not change often (e.g., scripts, static files, etc.)
COPY pretrained_model_wgt/infer_script.py /code

# Set the default command to run the infer script
CMD ["python", "infer_script.py"]
