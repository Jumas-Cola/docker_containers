FROM scipoptsuite/scipoptsuite:7.0.2

RUN mkdir -p src/notebooks
WORKDIR /src/notebooks

# Install python modules
RUN pip3 -q install pip --upgrade
RUN pip3 install jupyterlab pandas numpy scipy sympy matplotlib seaborn ipywidgets scikit-learn pymoo ortools

# Install Tini
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter-lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
