FROM crystallang/crystal:nightly-alpine-build

WORKDIR /app
ADD . /app

# TODO: When Crystal version 1.0 is official and dependencies are compliant, remove "--ignore-crystal-version"
RUN shards install --ignore-crystal-version
RUN shards update --ignore-crystal-version
RUN shards build
