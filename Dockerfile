FROM crystallang/crystal:nightly

WORKDIR /app
ADD . /app

RUN mkdir /app_tmp \
    && cd /app_tmp \
    && git clone https://github.com/crystal-community/icr.git \
    && cd icr \
    && make \
    && make test \
    && make install

RUN shards install
RUN shards update
RUN shards build
