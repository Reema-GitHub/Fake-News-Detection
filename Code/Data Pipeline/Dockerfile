# Instructions copied from - https://hub.docker.com/_/python/
FROM python:3-onbuild

# specify the port number the container should expose
EXPOSE 5000

COPY . /LuigiPipeline

WORKDIR /LuigiPipeline

RUN pip3 install nltk
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install luigi
RUN pip3 install unidecode
RUN pip3 install sklearn.preprocessing
RUN pip3 install sklearn.model_selection
RUN pip3 install sklearn.feature_extraction.text
RUN pip3 install sklearn
RUN pip3 install pandas_ml
RUN pip3 install matplotlib
RUN pip3 install sklearn.metrics

# run the command
CMD ["python", "./Luigi_implementation.py"]