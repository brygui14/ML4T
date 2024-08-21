FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# Set the working directory
WORKDIR /workspaces/ML4T/

# Install unzip utility (if not already installed)
RUN apt-get update && apt-get install -y unzip

# Copy environment.yml and zip files to the working directory
COPY environment.yml ./

# Create the conda environment from environment.yml
RUN conda env create -f environment.yml 

# Remove the environment.yml file if no longer needed
RUN rm environment.yml



# FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# # Set the working directory
# WORKDIR /workspaces/ML4T/

# # Copy environment.yml and other necessary files to the working directory
# COPY environment.yml ./
# COPY zips/ML4T_2024Fall.zip zips/assess_portfolio_2024Fall.zip ./zips/

# # Create the conda environment from environment.yml
# RUN conda env create -f environment.yml 

# # Remove the environment.yml file if no longer needed
# RUN rm environment.yml

# WORKDIR /workspaces/ML4T/

# ADD https://www.dropbox.com/scl/fi/puyw6qe5ybtiuhq4319r4/ML4T_2024Fall.zip?rlkey=mpfuugvfwz02losryev1m0kcv&st=mn3bji4f&dl=1 /tmp/zip/

# # ADD https://www.dropbox.com/scl/fi/zlg1wiwgb86690nciw5b9/assess_portfolio_2024Fall.zip?rlkey=96y7ml6lgoy34fh1f9n6eb6et&st=2iqb6pt8&dl=1 /tmp/zip/

# RUN sudo unzip /tmp/zip/ML4T_2024Fall.zip -d /tmp/zipextracted && sudo mv /tmp/zipextracted/* /workspaces/ML4T/

# # RUN sudo unzip /tmp/zip/assess_portfolio_2024Fall.zip -d /tmp/zipextracted/


# RUN sudo mv /tmp/zipextracted/* /workspaces/ML4T/

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
