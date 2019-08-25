require "./spec_helper"

describe Ai4cr do
  it "version in shard.yml matches version in Ai4cr::VERSION" do
    (`shards version .`).strip.should eq(Ai4cr::VERSION)
  end
end
