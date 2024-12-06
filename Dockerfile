FROM python:3.11

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the project files to the container
COPY ByteTrack /code/ByteTrack
COPY bytetrackCustom /code/bytetrackCustom
COPY config /code/config
COPY helpers /code/helpers
COPY lines_data/cam_line_data_3_3_2.csv /code/lines_data
COPY pretrained_model_wgt/infer_script.py /code

# Set the default command to run the infer script
CMD ["python", "infer_script.py"]

