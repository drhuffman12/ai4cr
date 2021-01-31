# FROM crystallang/crystal:nightly-build
FROM crystallang/crystal:nightly-alpine-build

WORKDIR /app
ADD . /app

# RUN shards install
# RUN shards update
# RUN shards build

RUN shards install --ignore-crystal-version
RUN shards update --ignore-crystal-version
RUN shards build