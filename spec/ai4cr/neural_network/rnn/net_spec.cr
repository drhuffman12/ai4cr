require "./../../../spec_helper"
require "json"

describe Ai4cr::NeuralNetwork::Rnn::Net do
  describe "#initialize" do
    describe "when given no params" do
      rnn_net = Ai4cr::NeuralNetwork::Rnn::Net.new

      expected_qty_time_cols = 5
      expected_qty_lpfc_layers = 3

      describe "exports to json as expected for" do
        File.write("tmp/rnn_net.json", rnn_net.to_pretty_json(indent: " "))

        contents = File.read("spec/data/neural_network/rnn/net/new.defaults.json")
        expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting
        actual_json = JSON.parse(rnn_net.to_json)
        
        describe "state" do
          it "config" do
            actual_json["state"]["config"].should eq(expected_json["state"]["config"])
          end
        end
      end
    end
  end
end