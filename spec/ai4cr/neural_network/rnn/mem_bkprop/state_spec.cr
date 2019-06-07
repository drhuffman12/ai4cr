require "./../../../../spec_helper"
require "json"

describe Ai4cr::NeuralNetwork::Rnn::MemBkprop::State do
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

      rnn_node_state = Ai4cr::NeuralNetwork::Rnn::MemBkprop::State.new(
        rnn_config,
        channel_set_index, channel_type, time_col_index,
        node_input_mappings
      )
      # File.write("tmp/rnn_node_state.json", rnn_node_state.to_pretty_json(indent: " "))

      describe "sets default values for" do
      end

      describe "exports to json as expected for" do
        contents = File.read("spec/data/neural_network/rnn/mem_bkprop/state/new.defaults.json")
        expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting
        actual_json = JSON.parse(rnn_node_state.to_json)
        
        it "config" do
          actual_json["config"].should eq(expected_json["config"])
        end

        it "recent_memory" do
          actual_json["recent_memory"].should eq(expected_json["recent_memory"])
        end

        describe "bp_net" do
          describe "state" do
            it "config" do
              actual_json["bp_net"]["state"]["config"].should eq(expected_json["bp_net"]["state"]["config"])
            end
          
            it "calculated_error_latest" do
              actual_json["bp_net"]["state"]["calculated_error_latest"].should eq(expected_json["bp_net"]["state"]["calculated_error_latest"])
            end
            
            it "track_history" do
              actual_json["bp_net"]["state"]["track_history"].should eq(expected_json["bp_net"]["state"]["track_history"])
            end
            
            describe "weights" do
              expected_weights = Array(Array(Array(Float64))).from_json(expected_json["bp_net"]["state"]["weights"].to_json)
              actual_weights = Array(Array(Array(Float64))).from_json(actual_json["bp_net"]["state"]["weights"].to_json)

              it ".size" do
                actual_weights.size.should eq(expected_weights.size)
              end

              it ".first.size" do
                actual_weights.first.size.should eq(expected_weights.first.size)
              end

              it ".first.first.size" do
                actual_weights.first.first.size.should eq(expected_weights.first.first.size)
              end
            end
            
            it "last_changes" do
              actual_json["bp_net"]["state"]["last_changes"].should eq(expected_json["bp_net"]["state"]["last_changes"])
            end
            
            it "activation_nodes" do
              actual_json["bp_net"]["state"]["activation_nodes"].should eq(expected_json["bp_net"]["state"]["activation_nodes"])
            end
            
            it "deltas" do
              actual_json["bp_net"]["state"]["deltas"].should eq(expected_json["bp_net"]["state"]["deltas"])
            end
            
            it "input_deltas" do
              actual_json["bp_net"]["state"]["input_deltas"].should eq(expected_json["bp_net"]["state"]["input_deltas"])
            end
            
            it "calculated_error_history" do
              actual_json["bp_net"]["state"]["calculated_error_history"].should eq(expected_json["bp_net"]["state"]["calculated_error_history"])
            end
          end
        end
      end
    end
  end
end