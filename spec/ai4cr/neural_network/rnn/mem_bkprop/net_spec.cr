require "./../../../../spec_helper"
require "json"

describe Ai4cr::NeuralNetwork::Rnn::MemBkprop::Net do
  describe "#initialize" do
    describe "when given no params" do
      rnn_config = Ai4cr::NeuralNetwork::Rnn::Config.new

      channel_set_index = 0
      channel_type = 0
      time_col_index = 0

      mem_bkprop_coord = {
        channel_set_index: 1,
        channel_type: 2,
        time_col_index: 3,
      }
      mem_bkprop_input_mappings = [mem_bkprop_coord]

      rnn_mem_bkprop_net = Ai4cr::NeuralNetwork::Rnn::MemBkprop::Net.new(
        rnn_config,
        channel_set_index, channel_type, time_col_index,
        mem_bkprop_input_mappings
      )
      File.write("tmp/rnn_mem_bkprop_net.json", rnn_mem_bkprop_net.to_pretty_json(indent: " "))

      describe "sets default values for" do
      end

      describe "exports to json as expected for" do
        contents = File.read("spec/data/neural_network/rnn/mem_bkprop/net/new.defaults.json")
        expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting
        actual_json = JSON.parse(rnn_mem_bkprop_net.to_json)
        
        describe "state" do
          it "config" do
            actual_json["state"]["config"].should eq(expected_json["state"]["config"])
          end
        end
      end
    end
  end
end