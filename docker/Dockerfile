FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV nnUNet_raw=/opt/nnunet/nnUNet_raw
ENV nnUNet_preprocessed=/opt/nnunet/nnUNet_preprocessed
ENV nnUNet_results=/opt/nnunet/nnUNet_results
ENV nnUNet_compile=F
RUN pip install --no-cache-dir nnunetv2==2.7.0
COPY nnUNetTrainerRCOversample.py /tmp/nnUNetTrainerRCOversample.py
RUN python -c "import nnunetv2,os,shutil;d=os.path.join(os.path.dirname(nnunetv2.__file__),'training','nnUNetTrainer','variants');shutil.copy('/tmp/nnUNetTrainerRCOversample.py',d);print('trainer installed ->',d)"
RUN python -c "from nnunetv2.utilities.find_class_by_name import recursive_find_python_class; import nnunetv2, os; from os.path import join; c=recursive_find_python_class(join(nnunetv2.__path__[0],'training','nnUNetTrainer'),'nnUNetTrainerRCOversample','nnunetv2.training.nnUNetTrainer'); assert c is not None, 'TRAINER NOT FOUND'; print('trainer resolves:',c)"
COPY nnUNet_results /opt/nnunet/nnUNet_results
RUN mkdir -p /opt/nnunet/nnUNet_raw /opt/nnunet/nnUNet_preprocessed
COPY predict.py /opt/predict.py
ENTRYPOINT ["python", "/opt/predict.py"]
