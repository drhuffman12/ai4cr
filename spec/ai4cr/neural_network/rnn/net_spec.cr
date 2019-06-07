require "./../../../spec_helper"
require "json"

describe Ai4cr::NeuralNetwork::Rnn::Net do
  describe "#initialize" do
    describe "when given no params" do
      rnn_net = Ai4cr::NeuralNetwork::Rnn::Net.new

      expected_qty_time_cols = 5
      expected_qty_lpfc_layers = 3

      describe "exports to json as expected for" do
        # File.write("tmp/rnn_net.json", rnn_net.to_pretty_json(indent: " "))

        contents = File.read("spec/data/neural_network/rnn/net/new.defaults.json")
        expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting
        actual_json = JSON.parse(rnn_net.to_json)
        
        describe "state" do
          it "config" do
            actual_json["state"]["config"].should eq(expected_json["state"]["config"])
          end
        end
      end

      it "exports to json and reimports from json" do
        Dir.mkdir("tmp") unless Dir.exists?("tmp")
        # File.write("tmp/rnn_net.json", rnn_net.to_pretty_json(indent: " "))
        tempfile_path = "tmp/rnn_net." + Time.utc.to_s("%Y-%m-%d.%H-%M-%S.%z") + "_" + rand.to_s + ".json"
        # tempfile = File.tempfile(tempfile_path)
        rnn_net_created = Ai4cr::NeuralNetwork::Rnn::Net.new
        rnn_net_created_json = rnn_net_created.to_json
        # File.write(tempfile_path, rnn_net_created.to_pretty_json(indent: " "))
        File.write(tempfile_path, rnn_net_created_json)

        contents = File.read(tempfile_path)
        rnn_net_loaded = Ai4cr::NeuralNetwork::Rnn::Net.from_json(contents)
        rnn_net_loaded_json = rnn_net_loaded.to_json

        expected_json = JSON.parse(rnn_net_created_json) # so can compare w/out human readable json file formatting
        actual_json = JSON.parse(rnn_net_loaded_json)
        
        expected_json.should eq(actual_json)
      ensure
        File.delete(tempfile_path) if tempfile_path && File.file?(tempfile_path)
      end
    end
  end
end