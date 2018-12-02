require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rnn::Net do
  describe "#initialize" do
    
    describe "with default params" do
      net = Ai4cr::NeuralNetwork::Rnn::Net.new
      defaults = {
        bias: true,
        dendrite_offsets: [1],
        hidden_layer_qty: 2,
        hidden_state_qty: 4,
        input_state_qty: 2,
        input_state_initial_values: [0.0, 0.0],
        memory_layer_qty: 2,
        output_state_qty: 3,
        output_winner_qty: 1,
        time_column_qty: 4
      }
      # default_nodes_hidden_channel = [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]

      
      describe "sets expected quantity for instance variable" do
        it "@bias" do
          net.bias.should eq(defaults[:bias])
        end

        it "@dendrite_offsets" do
          net.dendrite_offsets.should eq(defaults[:dendrite_offsets])
        end

        describe "@channel_input" do
          it "class" do
            net.channel_input.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Input)
          end

          it "@dendrite_offsets" do
            net.channel_input.dendrite_offsets.should eq(defaults[:dendrite_offsets])
          end

          describe "@node_sets" do
            it "class" do
              net.channel_input.node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
            end
  
            it "size" do
              net.channel_input.node_sets.size.should eq(defaults[:time_column_qty])
            end

            describe "first" do
              it "class" do
                net.channel_input.node_sets.first.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input)
              end

              it "state_qty" do
                net.channel_input.node_sets.first.state_qty.should eq(defaults[:input_state_qty])
              end

              it "state_range" do
                net.channel_input.node_sets.first.state_range.should eq((0..defaults[:input_state_qty]-1))
              end

              it "state_values" do
                net.channel_input.node_sets.first.state_values.should eq(defaults[:input_state_initial_values])
              end
            end
          end
  
          it "state_qty" do
            net.channel_input.state_qty.should eq(defaults[:input_state_qty])
          end

          it "time_column_qty" do
            net.channel_input.time_column_qty.should eq(defaults[:time_column_qty])
          end

          it "time_column_range" do
            net.channel_input.time_column_range.should eq((0..defaults[:time_column_qty]-1))
          end
  
        end

        describe "@channel_output" do
          it "class" do
            net.channel_output.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Output)
          end
        end

        it "@hidden_layer_qty" do
          net.hidden_layer_qty.should eq(defaults[:hidden_layer_qty])
        end

        describe "@hidden_layers" do
          describe "first" do
            it "class" do
              net.hidden_layers.first.should be_a(Ai4cr::NeuralNetwork::Rnn::HiddenLayer::First)
            end
          end

          # TODO: implement Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other then un-comment
          # it "size" do
          #   net.hidden_layers.size.should eq(defaults[:hidden_layer_qty])
          # end

          # TODO: implement Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other then un-comment
          # describe "second" do
          #   it "class" do
          #     net.hidden_layers.[1].should be_a(Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other)
          #   end
          # end
        end

        it "@hidden_state_qty" do
          net.hidden_state_qty.should eq(defaults[:hidden_state_qty])
        end

        it "@input_state_qty" do
          net.input_state_qty.should eq(defaults[:input_state_qty])
        end

        it "@memory_layer_qty" do
          net.memory_layer_qty.should eq(defaults[:memory_layer_qty])
        end

        it "@output_state_qty" do
          net.output_state_qty.should eq(defaults[:output_state_qty])
        end

        it "@output_winner_qty" do
          net.output_winner_qty.should eq(defaults[:output_winner_qty])
        end

        it "@time_column_qty" do
          net.time_column_qty.should eq(defaults[:time_column_qty])
        end
      end
    end
  end
end

