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
    udev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y v4l-utils


# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Copy only the requirements file to leverage caching
COPY requirements-locked.txt /code/requirements-locked.txt

# Install Python dependencies (this step will be cached until the requirements file changes)
RUN pip install --no-cache-dir -r /code/requirements-locked.txt

RUN groupadd -f video
RUN usermod -aG video root

# Copy the project files to the container (only necessary code)
COPY ByteTrack /code/ByteTrack
COPY bytetrackCustom /code/bytetrackCustom
COPY config /code/config
COPY helpers /code/helpers

# Copy the specific data file for caching purposes
COPY lines_data /code/lines_data/

# Copy any other files that do not change often (e.g., scripts, static files, etc.)
COPY infer_video_DEV.py /code

# Set the default command to run the infer script
CMD ["python", "infer_video_DEV.py"]
