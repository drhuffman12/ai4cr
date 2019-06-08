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

      describe "initializes @mem_bkprops" do
        describe "as an array (of channel sets)" do
          it "of expected size" do
            rnn_state.mem_bkprops.size.should eq(expected_qty_lpfc_layers)
          end
        end

        describe "containing an array (of channel types)" do
          it "of expected size" do
            expected_channel_size = 4
            rnn_state.mem_bkprops.first.size.should eq(expected_channel_size)
          end


          describe "containing an array (of time columns)" do
            it "of expected size" do
              rnn_state.mem_bkprops.first.first.size.should eq(expected_qty_time_cols)
            end
          end
        end
      end

      describe "exports to json as expected for" do
        # File.write("tmp/rnn_state.json", rnn_state.to_pretty_json(indent: " "))

        contents = File.read("spec/data/neural_network/rnn/state/new.defaults.json")
        expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting
        actual_json = JSON.parse(rnn_state.to_json)
        
        it "config" do
          actual_json["config"].should eq(expected_json["config"])
        end

        # describe "mem_bkprops" do
        #   expected_mem_bkprops = Array(Array(Array(Ai4cr::NeuralNetwork::Rnn::MemBkprop::Net))).from_json(expected_json["mem_bkprops"].to_json)
        #   actual_mem_bkprops = Array(Array(Array(Ai4cr::NeuralNetwork::Rnn::MemBkprop::Net))).from_json(actual_json["mem_bkprops"].to_json)

        #   it ".size" do
        #     actual_mem_bkprops.size.should eq(expected_mem_bkprops.size)
        #   end

        #   it ".first.size" do
        #     actual_mem_bkprops.first.size.should eq(expected_mem_bkprops.first.size)
        #   end

        #   it ".first.first.size" do
        #     actual_mem_bkprops.first.first.size.should eq(expected_mem_bkprops.first.first.size)
        #   end
        # end
        
      end
    end

    describe "when given params for a tiny rnn net" do
      rnn_state = Ai4cr::NeuralNetwork::Rnn::State.new(
        qty_states_in = 3,
        qty_states_channel_out = 4,
        qty_states_out = 2,
        qty_time_cols = 3,
        qty_lpfc_layers = 1,
        qty_hidden_laters = 0,
        qty_time_cols_neighbor_inputs = 1,
        qty_recent_memory = 1,
      )

      # File.write("tmp/rnn_state.tiny.json", rnn_state.to_pretty_json(indent: " "))
      # File.write("spec/data/neural_network/rnn/state/new.tiny.json", rnn_state.to_pretty_json(indent: " "))

      contents = File.read("spec/data/neural_network/rnn/state/new.tiny.json")
      expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting
      actual_json = JSON.parse(rnn_state.to_json)

      it "config" do
        actual_json["config"].should eq(expected_json["config"])
      end
    end
  end

  # describe "when given params for a large rnn net" do
  #   rnn_state = Ai4cr::NeuralNetwork::Rnn::State.new(
  #     qty_states_in = 80,
  #     qty_states_channel_out = 0,
  #     qty_states_out = 80,
  #     qty_time_cols = 40,
  #     qty_lpfc_layers = 8,
  #     qty_hidden_laters = 0,
  #     qty_time_cols_neighbor_inputs = 8,
  #     qty_recent_memory = 8,
  #   )

  #   File.write("tmp/rnn_state.large.json", rnn_state.to_pretty_json(indent: " "))
  #   # puts "sizeof(Ai4cr::NeuralNetwork::Rnn::State): #{sizeof(Ai4cr::NeuralNetwork::Rnn::State)}"
  #   # puts "instance_sizeof(Ai4cr::NeuralNetwork::Rnn::State): #{instance_sizeof(Ai4cr::NeuralNetwork::Rnn::State)}"
  #   # # puts "sizeof(rnn_state): #{sizeof(rnn_state)}"

  #   contents = File.read("spec/data/neural_network/rnn/state/new.large.json")
  #   expected_json = JSON.parse(contents) # so can compare w/out human readable json file formatting
  #   actual_json = JSON.parse(rnn_state.to_json)

  #   it "config" do
  #     actual_json["config"].should eq(expected_json["config"])
  #   end
  # end
end
