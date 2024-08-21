FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# Install necessary tools
RUN apt-get update && apt-get install -y git curl

# Clone the GitHub repository as the root user
WORKDIR /var/www/ml4t
RUN git clone https://github.com/brygui14/ML4T.git



# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda create -n my_new_env -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
