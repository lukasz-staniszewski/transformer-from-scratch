FROM nvcr.io/nvidia/pytorch:23.01-py3

ADD . /workspace/
WORKDIR /workspace/

RUN pip install -r requirements.txt 
CMD ["/bin/bash"]
