FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml environment.yml

RUN conda env create -f environment.yml 

RUN rm environment.yml

WORKDIR /workspaces/ML4T/

ADD https://www.dropbox.com/scl/fi/puyw6qe5ybtiuhq4319r4/ML4T_2024Fall.zip?rlkey=mpfuugvfwz02losryev1m0kcv&st=mn3bji4f&dl=1 /tmp/zip/

RUN sudo unzip /tmp/zip/ML4T_2024Fall.zip -d /tmp/zipextracted/

RUN sudo mv /tmp/zipextracted/* /workspaces/ML4T


# RUN unzip utilz.zip

# RUN rm environment.yml
# RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda create -n my_new_env -f /tmp/conda-tmp/environment.yml; fi \
#     && rm -rf /tmp/conda-tmp

# Copy all files from the parent directory (excluding the .git directory)
  # Adjust this if you want a different working directory
# COPY ../ .
# RUN git config --global core.excludesfile 

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
