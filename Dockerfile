FROM crystallang/crystal:nightly-alpine-build

WORKDIR /app
ADD . /app

RUN shards install
RUN shards update
RUN shards build
