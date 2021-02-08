FROM jupyter/datascience-notebook

USER root
RUN apt-get update && apt-get install -y graphviz

USER jovyan
RUN pip install graphviz
