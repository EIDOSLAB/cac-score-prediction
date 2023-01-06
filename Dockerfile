FROM eidos-service.di.unito.it/eidos-base-pytorch:1.10.0


RUN pip install pydicom
RUN pip install efficientnet-pytorch 
RUN pip install python-gdcm
RUN pip install pylibjpeg
RUN pip install seaborn
RUN pip install openpyxl

WORKDIR /src
COPY src /src 



RUN chmod 775 /src
RUN chown -R :1337 /src



ENTRYPOINT [ "python3"]
