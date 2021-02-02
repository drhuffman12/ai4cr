FROM crystallang/crystal:nightly-alpine-build

WORKDIR /app
ADD . /app

RUN shards install --ignore-crystal-version
RUN shards update --ignore-crystal-version
RUN shards build
