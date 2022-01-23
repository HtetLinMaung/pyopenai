FROM animcogn/face_recognition:cpu

# The rest of this file just runs an example script.

# If you wanted to use this Dockerfile to run your own app instead, maybe you would do this:
# COPY . /root/your_app_or_whatever
# RUN cd /root/your_app_or_whatever && \
#     pip3 install -r requirements.txt
# RUN whatever_command_you_run_to_start_your_app
WORKDIR /root/pyopenai

COPY . .

RUN pip3 install -r requirements.txt && \
    python3 setup.py install

CMD ["python", "main.py"]