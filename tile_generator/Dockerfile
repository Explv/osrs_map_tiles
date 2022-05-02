FROM python:3.10

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh -b -p /usr/local/bin/miniconda3
RUN ls /usr/local/bin/miniconda3/bin
ENV PATH="/usr/local/bin/miniconda3/bin:${PATH}"

# Create a conda env
RUN conda create --name vips python=3.7

# Update all future "RUN" commands to use the conda env
SHELL ["conda", "run", "--no-capture-output", "-n", "vips", "/bin/bash", "-c"]

# Install vips
RUN conda install --channel conda-forge pyvips

# Install python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install Java 11
RUN conda install -c conda-forge openjdk=11

# Install Maven
RUN conda install -c conda-forge maven

# Clone runelite repo
RUN git clone https://github.com/runelite/runelite.git /runelite

# Add required files
ADD src src

CMD [ "conda", "run", "--no-capture-output", "-n", "vips", "python", "./src/tile_generator.py" ]
