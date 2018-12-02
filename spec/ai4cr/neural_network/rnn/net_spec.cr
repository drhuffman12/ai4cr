require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rnn::Net do
  describe "#initialize" do
    describe "with default params" do
      net = Ai4cr::NeuralNetwork::Rnn::Net.new
      defaults = {
        hidden_layer_qty: 2,
        hidden_state_qty: 4,
        hidden_state_initial_values: [0.0, 0.0, 0.0, 0.0],
        
        time_column_qty: 4,
        bias: true,
        dendrite_offsets: [1],

        input_state_qty: 2,
        input_state_initial_values: [0.0, 0.0],

        memory_layer_qty: 2,

        output_state_qty: 3,
        output_state_initial_values: [0.0, 0.0, 0.0],
        output_winner_qty: 1
      }
          
      describe "creates the network structure" do
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


        # TODO: finish implementing Ai4cr::NeuralNetwork::Rnn::Channel::Output then un-comment
        describe "@channel_output" do
          it "class" do
            net.channel_output.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Output)
          end

          it "@dendrite_offsets" do
            net.channel_output.dendrite_offsets.should eq(defaults[:dendrite_offsets])
          end

          describe "@node_sets" do
            it "class" do
              net.channel_output.node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Output))
            end
  
            it "size" do
              net.channel_output.node_sets.size.should eq(defaults[:time_column_qty])
            end

            describe "first" do
              it "class" do
                net.channel_output.node_sets.first.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Output)
              end

              it "state_qty" do
                net.channel_output.node_sets.first.state_qty.should eq(defaults[:output_state_qty])
              end

              it "state_range" do
                net.channel_output.node_sets.first.state_range.should eq((0..defaults[:output_state_qty]-1))
              end

              it "state_values" do
                net.channel_output.node_sets.first.state_values.should eq(defaults[:output_state_initial_values])
              end
            end
          end

          it "state_qty" do
            net.channel_output.state_qty.should eq(defaults[:output_state_qty])
          end

          it "time_column_qty" do
            net.channel_output.time_column_qty.should eq(defaults[:time_column_qty])
          end

          it "time_column_range" do
            net.channel_output.time_column_range.should eq((0..defaults[:time_column_qty]-1))
          end

          # TODO: implement Ai4cr::NeuralNetwork::Rnn::WeightSet::Output then un-comment
          # describe "weights_output" do
          # end
        end

        it "@hidden_layer_qty" do
          net.hidden_layer_qty.should eq(defaults[:hidden_layer_qty])
        end

        describe "@hidden_layers" do
          it "size" do
            net.hidden_layers.size.should eq(defaults[:hidden_layer_qty])
          end

          describe "first" do
            it "class" do
              net.hidden_layers.first.should be_a(Ai4cr::NeuralNetwork::Rnn::HiddenLayer::First)
            end
          
            it "@bias" do
              net.hidden_layers.first.bias.should eq(defaults[:bias])
            end

            describe "channel_combo" do
              it "class" do
                net.hidden_layers.first.channel_combo.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Combo)
              end
            end

            describe "channel_future" do
              it "class" do
                net.hidden_layers.first.channel_future.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Future)
              end
            end

            describe "channel_local" do
              it "class" do
                net.hidden_layers.first.channel_local.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
              end
            end

            describe "channel_past" do
              it "class" do
                net.hidden_layers.first.channel_past.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Past)
              end
            end

            it "@dendrite_offsets" do
              net.hidden_layers.first.dendrite_offsets.should eq(defaults[:dendrite_offsets])
            end

            it "@is_first" do
              net.hidden_layers.first.is_first.should eq(true)
            end

            it "@layer_index" do
              net.hidden_layers.first.layer_index.should eq(0)
            end

            describe "previous_layer_output_channel" do
              it "class" do
                net.hidden_layers.first.previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Input)
              end
            end
  
            it "state_qty" do
              net.hidden_layers.first.state_qty.should eq(defaults[:hidden_state_qty])
            end
  
            it "time_column_qty" do
              net.hidden_layers.first.time_column_qty.should eq(defaults[:time_column_qty])
            end
  
            it "time_column_range" do
              net.hidden_layers.first.time_column_range.should eq((0..defaults[:time_column_qty]-1))
            end

            describe "weights_local" do
              it "class" do
                net.hidden_layers.first.weights_local.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input, Ai4cr::NeuralNetwork::Rnn::Channel::Input)))
              end

              it "size" do
                net.hidden_layers.first.weights_local.size.should eq(defaults[:time_column_qty])
              end

              # weights_local .. first
              describe "first" do
                it "class" do
                  net.hidden_layers.first.weights_local.first.should be_a(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input, Ai4cr::NeuralNetwork::Rnn::Channel::Input))
                end
  
                it "bias" do
                  net.hidden_layers.first.bias.should eq(defaults[:bias])
                end

                describe "center_input_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.center_input_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input)
                  end

                  it "state_qty" do
                    net.hidden_layers.first.weights_local.first.center_input_node_set.state_qty.should eq(defaults[:input_state_qty])
                  end
    
                  it "state_range" do
                    net.hidden_layers.first.weights_local.first.center_input_node_set.state_range.should eq((0..defaults[:input_state_qty]-1))
                  end
    
                  it "state_values" do
                    net.hidden_layers.first.weights_local.first.center_input_node_set.state_values.should eq(defaults[:input_state_initial_values])
                  end
                end
                
                it "@dendrite_offsets" do
                  net.hidden_layers.first.weights_local.first.dendrite_offsets.should eq(defaults[:dendrite_offsets])
                end  

                describe "output_channel" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
                  end
                end

                describe "output_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.output_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end
                end

                describe "previous_layer_output_channel" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Input)
                  end
                end

                describe "side_future_input_node_sets" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.side_future_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.first.side_future_input_node_sets.size.should eq(1)
                  end
                end

                describe "side_past_input_node_sets" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.side_past_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.first.side_past_input_node_sets.size.should eq(0)
                  end
                end
                
                it "@time_column_index" do
                  net.hidden_layers.first.weights_local.first.time_column_index.should eq(0)
                end  
                
                it "@time_column_index_future_offsets" do
                  net.hidden_layers.first.weights_local.first.time_column_index_future_offsets.should eq([1])
                end  
                
                it "@time_column_index_past_offsets" do
                  net.hidden_layers.first.weights_local.first.time_column_index_past_offsets.empty?.should be_true
                end  

                describe "weights_center_input_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.weights_center_input_node_set.should be_a(Array(Array(Float64)))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.first.weights_center_input_node_set.size.should eq(defaults[:hidden_state_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.first.weights_center_input_node_set.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                    end
                  end
                end  

                describe "weights_memory" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.weights_memory.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.first.weights_memory.size.should eq(defaults[:memory_layer_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.first.weights_memory.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.first.weights_memory.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_future" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.weights_side_future.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.first.weights_side_future.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.first.weights_side_future.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.first.weights_side_future.first.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                      end
                    end
                  end
                end

                describe "weights_side_past" do
                  it "class" do
                    net.hidden_layers.first.weights_local.first.weights_side_past.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.first.weights_side_past.empty?.should be_true
                  end
                end
              end

              # weights_local .. second
              describe "second" do
                it "class" do
                  net.hidden_layers.first.weights_local.[1].should be_a(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input, Ai4cr::NeuralNetwork::Rnn::Channel::Input))
                end
  
                it "bias" do
                  net.hidden_layers.first.bias.should eq(defaults[:bias])
                end

                describe "center_input_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].center_input_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input)
                  end

                  it "state_qty" do
                    net.hidden_layers.first.weights_local.[1].center_input_node_set.state_qty.should eq(defaults[:input_state_qty])
                  end
    
                  it "state_range" do
                    net.hidden_layers.first.weights_local.[1].center_input_node_set.state_range.should eq((0..defaults[:input_state_qty]-1))
                  end
    
                  it "state_values" do
                    net.hidden_layers.first.weights_local.[1].center_input_node_set.state_values.should eq(defaults[:input_state_initial_values])
                  end
                end
                
                it "@dendrite_offsets" do
                  net.hidden_layers.first.weights_local.[1].dendrite_offsets.should eq(defaults[:dendrite_offsets])
                end  

                describe "output_channel" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
                  end
                end

                describe "output_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].output_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end
                end

                describe "previous_layer_output_channel" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Input)
                  end
                end

                describe "side_future_input_node_sets" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].side_future_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[1].side_future_input_node_sets.size.should eq(1)
                  end
                end

                describe "side_past_input_node_sets" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].side_past_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[1].side_past_input_node_sets.size.should eq(1)
                  end
                end
                
                it "@time_column_index" do
                  net.hidden_layers.first.weights_local.[1].time_column_index.should eq(1)
                end  
                
                it "@time_column_index_future_offsets" do
                  net.hidden_layers.first.weights_local.[1].time_column_index_future_offsets.should eq([1])
                end  
                
                it "@time_column_index_past_offsets" do
                  net.hidden_layers.first.weights_local.[1].time_column_index_past_offsets.empty?.should be_false
                end  

                describe "weights_center_input_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].weights_center_input_node_set.should be_a(Array(Array(Float64)))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[1].weights_center_input_node_set.size.should eq(defaults[:hidden_state_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[1].weights_center_input_node_set.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                    end
                  end
                end  

                describe "weights_memory" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].weights_memory.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[1].weights_memory.size.should eq(defaults[:memory_layer_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[1].weights_memory.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.[1].weights_memory.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_future" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].weights_side_future.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[1].weights_side_future.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[1].weights_side_future.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.[1].weights_side_future.first.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                      end
                    end
                  end
                end

                describe "weights_side_past" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[1].weights_side_past.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[1].weights_side_past.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[1].weights_side_past.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.[1].weights_side_past.first.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                      end
                    end
                  end
                end
              end

              # weights_local .. third
              describe "third" do
                it "class" do
                  net.hidden_layers.first.weights_local.[2].should be_a(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input, Ai4cr::NeuralNetwork::Rnn::Channel::Input))
                end
  
                it "bias" do
                  net.hidden_layers.first.bias.should eq(defaults[:bias])
                end

                describe "center_input_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].center_input_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input)
                  end

                  it "state_qty" do
                    net.hidden_layers.first.weights_local.[2].center_input_node_set.state_qty.should eq(defaults[:input_state_qty])
                  end
    
                  it "state_range" do
                    net.hidden_layers.first.weights_local.[2].center_input_node_set.state_range.should eq((0..defaults[:input_state_qty]-1))
                  end
    
                  it "state_values" do
                    net.hidden_layers.first.weights_local.[2].center_input_node_set.state_values.should eq(defaults[:input_state_initial_values])
                  end
                end
                
                it "@dendrite_offsets" do
                  net.hidden_layers.first.weights_local.[2].dendrite_offsets.should eq(defaults[:dendrite_offsets])
                end  

                describe "output_channel" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
                  end
                end

                describe "output_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].output_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end
                end

                describe "previous_layer_output_channel" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Input)
                  end
                end

                describe "side_future_input_node_sets" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].side_future_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[2].side_future_input_node_sets.size.should eq(1)
                  end
                end

                describe "side_past_input_node_sets" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].side_past_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[2].side_past_input_node_sets.size.should eq(1)
                  end
                end
                
                it "@time_column_index" do
                  net.hidden_layers.first.weights_local.[2].time_column_index.should eq(2)
                end  
                
                it "@time_column_index_future_offsets" do
                  net.hidden_layers.first.weights_local.[2].time_column_index_future_offsets.should eq([1])
                end  
                
                it "@time_column_index_past_offsets" do
                  net.hidden_layers.first.weights_local.[2].time_column_index_past_offsets.empty?.should be_false
                end  

                describe "weights_center_input_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].weights_center_input_node_set.should be_a(Array(Array(Float64)))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[2].weights_center_input_node_set.size.should eq(defaults[:hidden_state_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[2].weights_center_input_node_set.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                    end
                  end
                end  

                describe "weights_memory" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].weights_memory.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[2].weights_memory.size.should eq(defaults[:memory_layer_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[2].weights_memory.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.[2].weights_memory.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_future" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].weights_side_future.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[2].weights_side_future.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[2].weights_side_future.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.[2].weights_side_future.first.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                      end
                    end
                  end
                end

                describe "weights_side_past" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[2].weights_side_past.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[2].weights_side_past.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[2].weights_side_past.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.[2].weights_side_past.first.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                      end
                    end
                  end
                end
              end

              # weights_local .. fourth
              describe "fourth" do
                it "class" do
                  net.hidden_layers.first.weights_local.[3].should be_a(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input, Ai4cr::NeuralNetwork::Rnn::Channel::Input))
                end
  
                it "bias" do
                  net.hidden_layers.first.bias.should eq(defaults[:bias])
                end

                describe "center_input_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].center_input_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input)
                  end

                  it "state_qty" do
                    net.hidden_layers.first.weights_local.[3].center_input_node_set.state_qty.should eq(defaults[:input_state_qty])
                  end
    
                  it "state_range" do
                    net.hidden_layers.first.weights_local.[3].center_input_node_set.state_range.should eq((0..defaults[:input_state_qty]-1))
                  end
    
                  it "state_values" do
                    net.hidden_layers.first.weights_local.[3].center_input_node_set.state_values.should eq(defaults[:input_state_initial_values])
                  end
                end
                
                it "@dendrite_offsets" do
                  net.hidden_layers.first.weights_local.[3].dendrite_offsets.should eq(defaults[:dendrite_offsets])
                end  

                describe "output_channel" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
                  end
                end

                describe "output_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].output_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end
                end

                describe "previous_layer_output_channel" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Input)
                  end
                end

                describe "side_future_input_node_sets" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].side_future_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[3].side_future_input_node_sets.size.should eq(0)
                  end
                end

                describe "side_past_input_node_sets" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].side_past_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Input))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[3].side_past_input_node_sets.size.should eq(1)
                  end
                end
                
                it "@time_column_index" do
                  net.hidden_layers.first.weights_local.[3].time_column_index.should eq(3)
                end  
                
                it "@time_column_index_future_offsets" do
                  net.hidden_layers.first.weights_local.[3].time_column_index_future_offsets.empty?.should be_true
                end  
                
                it "@time_column_index_past_offsets" do
                  net.hidden_layers.first.weights_local.[3].time_column_index_past_offsets.should eq([-1])
                end  

                describe "weights_center_input_node_set" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].weights_center_input_node_set.should be_a(Array(Array(Float64)))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[3].weights_center_input_node_set.size.should eq(defaults[:hidden_state_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[3].weights_center_input_node_set.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                    end
                  end
                end  

                describe "weights_memory" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].weights_memory.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[3].weights_memory.size.should eq(defaults[:memory_layer_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[3].weights_memory.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.[3].weights_memory.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_future" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].weights_side_future.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[3].weights_side_future.size.should eq(0)
                  end
                end

                describe "weights_side_past" do
                  it "class" do
                    net.hidden_layers.first.weights_local.[3].weights_side_past.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.first.weights_local.[3].weights_side_past.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.first.weights_local.[3].weights_side_past.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.first.weights_local.[3].weights_side_past.first.first.size.should eq(defaults[:input_state_qty] + (defaults[:bias] ? 1 : 0))
                      end
                    end
                  end
                end
              end
            end

            # TODO: implement Ai4cr::NeuralNetwork::Rnn::WeightSet::Past then un-comment
            # describe "weights_past" do
            # end

            # TODO: implement Ai4cr::NeuralNetwork::Rnn::WeightSet::Future then un-comment
            # describe "weights_future" do
            # end

            # TODO: implement Ai4cr::NeuralNetwork::Rnn::WeightSet::Combo then un-comment
            # describe "weights_combo" do
            # end
          end

          describe "second" do
            it "class" do
              net.hidden_layers.[1].should be_a(Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other)
            end

            # TODO: implement Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other then un-comment
            # describe "(etc)" do
            # end
          end

          describe "last" do
            it "class" do
              net.hidden_layers.last.should be_a(Ai4cr::NeuralNetwork::Rnn::HiddenLayer::Other)
            end
          
            it "@bias" do
              net.hidden_layers.last.bias.should eq(false)
            end

            describe "channel_combo" do
              it "class" do
                net.hidden_layers.last.channel_combo.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Combo)
              end
            end

            describe "channel_future" do
              it "class" do
                net.hidden_layers.last.channel_future.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Future)
              end
            end

            describe "channel_local" do
              it "class" do
                net.hidden_layers.last.channel_local.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
              end
            end

            describe "channel_past" do
              it "class" do
                net.hidden_layers.last.channel_past.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Past)
              end
            end

            it "@dendrite_offsets" do
              net.hidden_layers.last.dendrite_offsets.should eq(defaults[:dendrite_offsets])
            end

            it "@is_first" do
              net.hidden_layers.last.is_first.should eq(false)
            end

            it "@layer_index" do
              net.hidden_layers.last.layer_index.should eq(defaults[:hidden_layer_qty] - 1)
            end

            describe "previous_layer_output_channel" do
              it "class" do
                net.hidden_layers.last.previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Combo)
              end
            end
  
            it "state_qty" do
              net.hidden_layers.last.state_qty.should eq(defaults[:hidden_state_qty])
            end
  
            it "time_column_qty" do
              net.hidden_layers.last.time_column_qty.should eq(defaults[:time_column_qty])
            end
  
            it "time_column_range" do
              net.hidden_layers.last.time_column_range.should eq((0..defaults[:time_column_qty]-1))
            end

            describe "weights_local" do
              it "class" do
                net.hidden_layers.last.weights_local.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden, Ai4cr::NeuralNetwork::Rnn::Channel::Combo)))
              end

              it "size" do
                net.hidden_layers.last.weights_local.size.should eq(defaults[:time_column_qty])
              end

              # weights_local .. first
              describe "first" do
                it "class" do
                  net.hidden_layers.last.weights_local.first.should be_a(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden, Ai4cr::NeuralNetwork::Rnn::Channel::Combo))
                end
  
                it "bias" do
                  net.hidden_layers.last.bias.should eq(false)
                end

                describe "center_input_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.center_input_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end

                  it "state_qty" do
                    net.hidden_layers.last.weights_local.first.center_input_node_set.state_qty.should eq(defaults[:hidden_state_qty])
                  end
    
                  it "state_range" do
                    net.hidden_layers.last.weights_local.first.center_input_node_set.state_range.should eq((0..defaults[:hidden_state_qty]-1))
                  end
    
                  it "state_values" do
                    net.hidden_layers.last.weights_local.first.center_input_node_set.state_values.should eq(defaults[:hidden_state_initial_values])
                  end
                end
                
                it "@dendrite_offsets" do
                  net.hidden_layers.last.weights_local.first.dendrite_offsets.should eq(defaults[:dendrite_offsets])
                end  

                describe "output_channel" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
                  end
                end

                describe "output_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.output_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end
                end

                describe "previous_layer_output_channel" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Combo)
                  end
                end

                describe "side_future_input_node_sets" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.side_future_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.first.side_future_input_node_sets.size.should eq(1)
                  end
                end

                describe "side_past_input_node_sets" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.side_past_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.first.side_past_input_node_sets.size.should eq(0)
                  end
                end
                
                it "@time_column_index" do
                  net.hidden_layers.last.weights_local.first.time_column_index.should eq(0)
                end  
                
                it "@time_column_index_future_offsets" do
                  net.hidden_layers.last.weights_local.first.time_column_index_future_offsets.should eq([1])
                end  
                
                it "@time_column_index_past_offsets" do
                  net.hidden_layers.last.weights_local.first.time_column_index_past_offsets.empty?.should be_true
                end  

                describe "weights_center_input_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.weights_center_input_node_set.should be_a(Array(Array(Float64)))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.first.weights_center_input_node_set.size.should eq(defaults[:hidden_state_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.first.weights_center_input_node_set.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  end
                end  

                describe "weights_memory" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.weights_memory.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.first.weights_memory.size.should eq(defaults[:memory_layer_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.first.weights_memory.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.first.weights_memory.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_future" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.weights_side_future.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.first.weights_side_future.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.first.weights_side_future.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.first.weights_side_future.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_past" do
                  it "class" do
                    net.hidden_layers.last.weights_local.first.weights_side_past.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.first.weights_side_past.empty?.should be_true
                  end
                end
              end

              # weights_local .. second
              describe "second" do
                it "class" do
                  net.hidden_layers.last.weights_local.[1].should be_a(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden, Ai4cr::NeuralNetwork::Rnn::Channel::Combo))
                end
  
                it "bias" do
                  net.hidden_layers.last.bias.should eq(false)
                end

                describe "center_input_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].center_input_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end

                  it "state_qty" do
                    net.hidden_layers.last.weights_local.[1].center_input_node_set.state_qty.should eq(defaults[:hidden_state_qty])
                  end
    
                  it "state_range" do
                    net.hidden_layers.last.weights_local.[1].center_input_node_set.state_range.should eq((0..defaults[:hidden_state_qty]-1))
                  end
    
                  it "state_values" do
                    net.hidden_layers.last.weights_local.[1].center_input_node_set.state_values.should eq(defaults[:hidden_state_initial_values])
                  end
                end
                
                it "@dendrite_offsets" do
                  net.hidden_layers.last.weights_local.[1].dendrite_offsets.should eq(defaults[:dendrite_offsets])
                end  

                describe "output_channel" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
                  end
                end

                describe "output_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].output_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end
                end

                describe "previous_layer_output_channel" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Combo)
                  end
                end

                describe "side_future_input_node_sets" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].side_future_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[1].side_future_input_node_sets.size.should eq(1)
                  end
                end

                describe "side_past_input_node_sets" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].side_past_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[1].side_past_input_node_sets.size.should eq(1)
                  end
                end
                
                it "@time_column_index" do
                  net.hidden_layers.last.weights_local.[1].time_column_index.should eq(1)
                end  
                
                it "@time_column_index_future_offsets" do
                  net.hidden_layers.last.weights_local.[1].time_column_index_future_offsets.should eq([1])
                end  
                
                it "@time_column_index_past_offsets" do
                  net.hidden_layers.last.weights_local.[1].time_column_index_past_offsets.empty?.should be_false
                end  

                describe "weights_center_input_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].weights_center_input_node_set.should be_a(Array(Array(Float64)))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[1].weights_center_input_node_set.size.should eq(defaults[:hidden_state_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[1].weights_center_input_node_set.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  end
                end  

                describe "weights_memory" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].weights_memory.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[1].weights_memory.size.should eq(defaults[:memory_layer_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[1].weights_memory.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.[1].weights_memory.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_future" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].weights_side_future.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[1].weights_side_future.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[1].weights_side_future.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.[1].weights_side_future.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_past" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[1].weights_side_past.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[1].weights_side_past.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[1].weights_side_past.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.[1].weights_side_past.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end
              end

              # weights_local .. third
              describe "third" do
                it "class" do
                  net.hidden_layers.last.weights_local.[2].should be_a(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden, Ai4cr::NeuralNetwork::Rnn::Channel::Combo))
                end
  
                it "bias" do
                  net.hidden_layers.last.bias.should eq(false)
                end

                describe "center_input_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].center_input_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end

                  it "state_qty" do
                    net.hidden_layers.last.weights_local.[2].center_input_node_set.state_qty.should eq(defaults[:hidden_state_qty])
                  end
    
                  it "state_range" do
                    net.hidden_layers.last.weights_local.[2].center_input_node_set.state_range.should eq((0..defaults[:hidden_state_qty]-1))
                  end
    
                  it "state_values" do
                    net.hidden_layers.last.weights_local.[2].center_input_node_set.state_values.should eq(defaults[:hidden_state_initial_values])
                  end
                end
                
                it "@dendrite_offsets" do
                  net.hidden_layers.last.weights_local.[2].dendrite_offsets.should eq(defaults[:dendrite_offsets])
                end  

                describe "output_channel" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
                  end
                end

                describe "output_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].output_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end
                end

                describe "previous_layer_output_channel" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Combo)
                  end
                end

                describe "side_future_input_node_sets" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].side_future_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[2].side_future_input_node_sets.size.should eq(1)
                  end
                end

                describe "side_past_input_node_sets" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].side_past_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[2].side_past_input_node_sets.size.should eq(1)
                  end
                end
                
                it "@time_column_index" do
                  net.hidden_layers.last.weights_local.[2].time_column_index.should eq(2)
                end  
                
                it "@time_column_index_future_offsets" do
                  net.hidden_layers.last.weights_local.[2].time_column_index_future_offsets.should eq([1])
                end  
                
                it "@time_column_index_past_offsets" do
                  net.hidden_layers.last.weights_local.[2].time_column_index_past_offsets.empty?.should be_false
                end  

                describe "weights_center_input_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].weights_center_input_node_set.should be_a(Array(Array(Float64)))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[2].weights_center_input_node_set.size.should eq(defaults[:hidden_state_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[2].weights_center_input_node_set.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  end
                end  

                describe "weights_memory" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].weights_memory.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[2].weights_memory.size.should eq(defaults[:memory_layer_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[2].weights_memory.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.[2].weights_memory.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_future" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].weights_side_future.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[2].weights_side_future.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[2].weights_side_future.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.[2].weights_side_future.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_past" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[2].weights_side_past.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[2].weights_side_past.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[2].weights_side_past.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.[2].weights_side_past.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end
              end

              # weights_local .. fourth
              describe "fourth" do
                it "class" do
                  net.hidden_layers.last.weights_local.[3].should be_a(Ai4cr::NeuralNetwork::Rnn::WeightSet::Local(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden, Ai4cr::NeuralNetwork::Rnn::Channel::Combo))
                end
  
                it "bias" do
                  net.hidden_layers.last.bias.should eq(false)
                end

                describe "center_input_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].center_input_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end

                  it "state_qty" do
                    net.hidden_layers.last.weights_local.[3].center_input_node_set.state_qty.should eq(defaults[:hidden_state_qty])
                  end
    
                  it "state_range" do
                    net.hidden_layers.last.weights_local.[3].center_input_node_set.state_range.should eq((0..defaults[:hidden_state_qty]-1))
                  end
    
                  it "state_values" do
                    net.hidden_layers.last.weights_local.[3].center_input_node_set.state_values.should eq(defaults[:hidden_state_initial_values])
                  end
                end
                
                it "@dendrite_offsets" do
                  net.hidden_layers.last.weights_local.[3].dendrite_offsets.should eq(defaults[:dendrite_offsets])
                end  

                describe "output_channel" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Local)
                  end
                end

                describe "output_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].output_node_set.should be_a(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden)
                  end
                end

                describe "previous_layer_output_channel" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].previous_layer_output_channel.should be_a(Ai4cr::NeuralNetwork::Rnn::Channel::Combo)
                  end
                end

                describe "side_future_input_node_sets" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].side_future_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[3].side_future_input_node_sets.size.should eq(0)
                  end
                end

                describe "side_past_input_node_sets" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].side_past_input_node_sets.should be_a(Array(Ai4cr::NeuralNetwork::Rnn::NodeSet::Hidden))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[3].side_past_input_node_sets.size.should eq(1)
                  end
                end
                
                it "@time_column_index" do
                  net.hidden_layers.last.weights_local.[3].time_column_index.should eq(3)
                end  
                
                it "@time_column_index_future_offsets" do
                  net.hidden_layers.last.weights_local.[3].time_column_index_future_offsets.empty?.should be_true
                end  
                
                it "@time_column_index_past_offsets" do
                  net.hidden_layers.last.weights_local.[3].time_column_index_past_offsets.should eq([-1])
                end  

                describe "weights_center_input_node_set" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].weights_center_input_node_set.should be_a(Array(Array(Float64)))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[3].weights_center_input_node_set.size.should eq(defaults[:hidden_state_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[3].weights_center_input_node_set.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  end
                end  

                describe "weights_memory" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].weights_memory.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[3].weights_memory.size.should eq(defaults[:memory_layer_qty])
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[3].weights_memory.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.[3].weights_memory.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end

                describe "weights_side_future" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].weights_side_future.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[3].weights_side_future.size.should eq(0)
                  end
                end

                describe "weights_side_past" do
                  it "class" do
                    net.hidden_layers.last.weights_local.[3].weights_side_past.should be_a(Array(Array(Array(Float64))))
                  end
                  
                  it "size" do
                    net.hidden_layers.last.weights_local.[3].weights_side_past.size.should eq(1)
                  end
                  
                  describe "first" do
                    it "size" do
                      net.hidden_layers.last.weights_local.[3].weights_side_past.first.size.should eq(defaults[:hidden_state_qty])
                    end
                  
                    describe "first" do
                      it "size" do
                        net.hidden_layers.last.weights_local.[3].weights_side_past.first.first.size.should eq(defaults[:hidden_state_qty])
                      end
                    end
                  end
                end
              end
            end

            # TODO: implement Ai4cr::NeuralNetwork::Rnn::WeightSet::Past then un-comment
            # describe "weights_past" do
            # end

            # TODO: implement Ai4cr::NeuralNetwork::Rnn::WeightSet::Future then un-comment
            # describe "weights_future" do
            # end

            # TODO: implement Ai4cr::NeuralNetwork::Rnn::WeightSet::Combo then un-comment
            # describe "weights_combo" do
            # end
          end
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
