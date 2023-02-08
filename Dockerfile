FROM public.ecr.aws/g7a8t7v6/jobs-container-keras-export-base:5db927fed029b8725cd7be77753bca6acb600438

WORKDIR /scripts

# Install extra dependencies here
COPY requirements.txt ./
RUN /app/keras/.venv/bin/pip3 install --no-cache-dir -r requirements.txt

# Copy all files to the home directory
COPY . ./

# The train command (we run this from the keras venv, which has all dependencies)
ENTRYPOINT [ "./run-python-with-venv.sh", "keras", "train.py" ]
