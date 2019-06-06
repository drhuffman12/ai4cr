require "./../../../spec_helper"
require "json"

describe Ai4cr::NeuralNetwork::Rnn::State do
  describe "#initialize" do
    describe "when given no params" do
      rnn_state = Ai4cr::NeuralNetwork::Rnn::State.new

      expected_qty_time_cols = 5
      expected_qty_lpfc_layers = 3

      describe "sets default values for" do
        it "qty_states_in" do
          expected_qty_states_in = 3
          rnn_state.config.qty_states_in.should eq(expected_qty_states_in)
        end
  
        it "qty_states_out" do
          expected_qty_states_out = 4
          rnn_state.config.qty_states_out.should eq(expected_qty_states_out)
        end
  
        it "qty_time_cols" do
          rnn_state.config.qty_time_cols.should eq(expected_qty_time_cols)
        end
  
        it "qty_lpfc_layers" do
          rnn_state.config.qty_lpfc_layers.should eq(expected_qty_lpfc_layers)
        end
  
        it "qty_recent_memory" do
          expected_qty_recent_memory = 2
          rnn_state.config.qty_recent_memory.should eq(expected_qty_recent_memory)
        end
  
        # it "structure_hidden_laters" do
        #   expected_structure_hidden_laters = [] of Int32
        #   rnn_state.config.structure_hidden_laters.should eq(expected_structure_hidden_laters)
        # end
  
        # it "disable_bias" do
        #   expected_disable_bias = true
        #   rnn_state.disable_bias.should eq(expected_disable_bias)
        # end
  
        it "learning_rate" do
          expected_learning_rate = 0.25
          rnn_state.config.learning_rate.should eq(expected_learning_rate)
        end
  
        it "momentum" do
          expected_momentum = 0.1
          rnn_state.config.momentum.should eq(expected_momentum)
        end
      end

      describe "initializes @nodes" do
        describe "as an array (of channel sets)" do
          it "of expected size" do
            rnn_state.nodes.size.should eq(expected_qty_lpfc_layers)
          end
        end

        describe "containing an array (of channel types)" do
          it "of expected size" do
            expected_channel_size = 4
            rnn_state.nodes.first.size.should eq(expected_channel_size)
          end


          describe "containing an array (of time columns)" do
            it "of expected size" do
              rnn_state.nodes.first.first.size.should eq(expected_qty_time_cols)
            end
          end
        end
      end

      # it "exports to json as expected" do
      #   File.write("tmp/rnn_state.json", rnn_state.to_pretty_json(indent: " "))

      #   puts
      #   puts "rnn_state:"
      #   puts rnn_state.to_json
      #   puts

      #   contents = File.read("spec/data/neural_network/rnn/state/new.defaults.json")
      #   expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting

      #   JSON.parse(rnn_state.to_json).should eq(expected_json)
      # end
    end
  end
end