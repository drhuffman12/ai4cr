require "./../../../../spec_helper"
require "json"

describe Ai4cr::NeuralNetwork::Rnn::MemBkprop::Config do
  describe "#initialize" do
    describe "when given no params" do
      rnn_config = Ai4cr::NeuralNetwork::Rnn::Config.new

      channel_set_index = 0
      channel_type = 0
      time_col_index = 0

      node_coord = {
        channel_set_index: 1,
        channel_type: 2,
        time_col_index: 3,
      }
      node_input_mappings = [node_coord]

      rnn_node_config = Ai4cr::NeuralNetwork::Rnn::MemBkprop::Config.new(
        rnn_config,
        channel_set_index, channel_type, time_col_index,
        node_input_mappings
      )
      # File.write("tmp/rnn_node_config.json", rnn_node_config.to_pretty_json(indent: " "))

      it "exports to json as expected" do
        contents = File.read("spec/data/neural_network/rnn/mem_bkprop/config/new.defaults.json")
        expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting

        JSON.parse(rnn_node_config.to_json).should eq(expected_json)
      end
    end
  end
end