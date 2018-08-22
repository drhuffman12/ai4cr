FROM crystallang/crystal:nightly

WORKDIR /app
ADD . /app

RUN shards install
RUN shards update
RUN shards build
