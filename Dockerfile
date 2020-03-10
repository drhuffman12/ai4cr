FROM crystallang/crystal:nightly

RUN apt update && apt install tree

WORKDIR /app
ADD . /app

RUN shards install
RUN shards update
RUN shards build
