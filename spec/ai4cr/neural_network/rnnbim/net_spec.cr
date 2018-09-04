require "./../../../spec_helper"

describe Ai4cr::NeuralNetwork::Rnnbim::Net do
  describe "with default params" do
    net = Ai4cr::NeuralNetwork::Rnnbim::Net.new

    expected_to_channel_keys_for_hidden_weights = [:past, :local, :future, :combo]

    default_expected_input_state_qty = 4
    default_expected_output_state_qty = 2
    default_expected_hidden_layer_qty = 2
    default_expected_time_column_qty = 2
    default_expected_hidden_layer_scale = 1.0
    default_expected_hidden_state_qty = 3 # (avg of in & out)*scale

    default_expected_time_column_range = (0..7)
    default_expected_input_state_range = (0..3) # (0..default_expected_input_state_qty-1)
    default_expected_hidden_state_range = (0..2) # (0..default_expected_hidden_state_qty-1)
    default_expected_output_state_range = (0..1) # (0..default_expected_output_state_qty-1)

    default_expected_hidden_offset_scales = [1,2]

    default_expected_nodes_hidden = [
      {
        :current => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        },
        :mem => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      } #,
        # :mem_same_image => {
        #   :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        # },
        # :mem_after_image => {
        #   :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        # }
      },
      {
        :current => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        },
        :mem => {
          :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        } # ,
        # :mem_same_image => {
        #   :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        # },
        # :mem_after_image => {
        #   :past => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :local => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :future => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        #   :combo => [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        # }
      }
    ]
    
    default_expected_meta = Ai4cr::NeuralNetwork::Rnnbim::NetMeta.new(
      default_expected_time_column_range,
      default_expected_hidden_state_range,
      default_expected_input_state_range,
      default_expected_output_state_range
    )
    default_expected_layer_names = ["hidden_0", "hidden_1", "output"]
    default_expected_meta_weights = default_expected_meta.weights
    
    describe "#initialize" do
      
      describe "sets expected quantity for instance variable" do
        it "@input_state_qty" do
          net.input_state_qty.should eq(default_expected_input_state_qty)
        end
  
        it "@hidden_state_qty" do
          net.hidden_state_qty.should eq(default_expected_hidden_state_qty)
        end
  
        it "@output_state_qty" do
          net.output_state_qty.should eq(default_expected_output_state_qty)
        end
  
        it "@hidden_layer_qty" do
          net.hidden_layer_qty.should eq(default_expected_hidden_layer_qty)
        end
  
        it "@hidden_layer_scale" do
          net.hidden_layer_scale.should eq(default_expected_hidden_layer_scale)
        end
  
        it "@hidden_offset_scales" do
          net.hidden_offset_scales.should eq(default_expected_hidden_offset_scales)
        end
      end

      describe "sets expected range for instance variable" do
        it "@input_state_range" do
          net.input_state_range.should eq(default_expected_input_state_range)
        end
  
        it "@hidden_state_range" do
          net.hidden_state_range.should eq(default_expected_hidden_state_range)
        end
  
        it "@output_state_range" do
          net.output_state_range.should eq(default_expected_output_state_range)
        end
      end

      describe "sets expected array structure for" do
        default_expected_nodes_in = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        default_expected_nodes_out = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
  
        it "@nodes_in" do
          net.nodes_in.should eq(default_expected_nodes_in)
        end
          
        it "#nodes_out" do
          net.nodes_out.should eq(default_expected_nodes_out)
        end
          
        it "#delta_out" do
          net.delta_out.should eq(default_expected_nodes_out)
        end
      end

      describe "sets expected hash structure for" do
        it "@nodes_hidden" do
          net.nodes_hidden.should eq(default_expected_nodes_hidden)
        end

        it "@delta_hidden" do
          net.delta_hidden.should eq(default_expected_nodes_hidden)
        end
      end
    end

    describe "#init_hidden_nodes" do
      it "returns expected results" do
        net.init_hidden_nodes.should eq(default_expected_nodes_hidden)
      end
    end

    describe "#init_network_weights" do
      weights = net.init_network_weights

      it "has expected top level 'layer' keys" do
        weights.keys.should eq(default_expected_meta_weights.keys)
        weights.keys.should eq(default_expected_layer_names)
      end

      default_expected_meta_weights.keys.each do |layer_name|
        describe "for top level 'layer' name \"#{layer_name}\"" do
          it "has expected 'to channel' keys" do
            weights[layer_name].keys.should eq(default_expected_meta_weights[layer_name].keys)
          end

          default_expected_meta_weights[layer_name].each do |to_channel_key, to_channel_meta|
            describe "for 'to channel' key :#{to_channel_key} has an array (aka time column indexes)" do
              expected_size = to_channel_meta[:chrono_size].as(Int32)

              it "of expected size" do
                weights[layer_name][to_channel_key].size.should eq(expected_size)
              end

              [:time_col_first_keys, :time_col_mid_keys, :time_col_last_keys].each do |time_col_section_key|
                describe "with a time column section key (:#{time_col_section_key}) which" do
                  time_col_index = case time_col_section_key
                    when :time_col_first_keys
                      0
                    when :time_col_last_keys
                      -1
                    else
                      expected_size / 2
                    end

                  keys = weights[layer_name][to_channel_key][time_col_index].keys.sort
                  expected_from_channel_meta = to_channel_meta[time_col_section_key].as(Hash(Symbol, Hash(Symbol, Int32)))
                  expected_from_channel_keys = expected_from_channel_meta.keys.sort
                  # expected_from_channel_keys = to_channel_meta[time_col_section_key].as(Hash(MetaChronoKey, MetaWeightsFromChannel)).keys.sort

                  it "has expected 'from channel' keys" do
                    # keys = weights[layer_name][to_channel_key].first.keys.sort

                    # puts
                    # puts "keys: #{keys}"
                    # puts
                    
                    keys.should eq(expected_from_channel_keys) # [:time_col_first_keys])
                  end

                  expected_from_channel_keys.each do |expected_from_channel_key|

                    expected_size_meta = expected_from_channel_meta[expected_from_channel_key]

                    describe "for 'from channel' key :#{expected_from_channel_key}, it defines simple weights" do
                      # puts
                      # puts "to_channel_meta: #{to_channel_meta}"
                      # puts
                      # puts "expected_from_channel_meta: #{expected_from_channel_meta}"
                      # puts
                      # puts "expected_from_channel_keys: #{expected_from_channel_keys}"
                      # puts

                      it "with expected :in_size" do
                        expected_in_size = expected_size_meta[:in_size]
                        ins = weights[layer_name][to_channel_key][time_col_index][expected_from_channel_key].as(Array(Array(Float64)))
                        in_size = ins.size
                        # in_size = weights[layer_name][to_channel_key][time_col_index].first.size

                        # puts
                        # puts "layer_name: '#{layer_name}', to_channel_key: :#{to_channel_key}, time_col_index: #{time_col_index}, ins: #{ins}, expected_from_channel_key: :#{expected_from_channel_key}, in_size: #{in_size}, expected_in_size: #{expected_in_size}"
                        # puts
                        
                        in_size.should eq(expected_in_size)
                      end
    
                      it "with expected :out_size" do
                        expected_out_size = expected_size_meta[:out_size]
                        outs = weights[layer_name][to_channel_key][time_col_index][expected_from_channel_key].as(Array(Array(Float64)))
                        out_size = outs.first.size
                        # out_size = weights[layer_name][to_channel_key][time_col_index].size
                        out_size.should eq(expected_out_size)
                      end
                    end
                  end
                end
              end

              # describe "with a mid element (:time_col_mid_keys) which" do
              #   it "has expected keys" do
              #     # weights[layer_name][to_channel_key].last.should eq("TBD")
              #     mid_index = weights[layer_name][to_channel_key].size / 2
              #     keys = weights[layer_name][to_channel_key][mid_index].keys.sort
              #     expected_keys = to_channel_meta[:time_col_mid_keys].as(Hash(Symbol, Hash(Symbol, Int32))).keys.sort
              #     keys.should eq(expected_keys) # [:time_col_mid_keys])
              #   end
              # end

              # describe "with a last element (:time_col_last_keys) which" do
              #   it "has expected keys" do
              #     # weights[layer_name][to_channel_key].last.should eq("TBD")
              #     keys = weights[layer_name][to_channel_key].last.keys.sort
              #     expected_keys = to_channel_meta[:time_col_last_keys].as(Hash(Symbol, Hash(Symbol, Int32))).keys.sort
              #     keys.should eq(expected_keys) # [:time_col_last_keys])
              #   end
              # end

            end
          end

        end
      end

      # it "DEBUG output values" do
      #   weights["output"].should eq([0.0]) # TODO: for debugging; remove before merging to master
      # end

      # it "DEBUG hidden_0 values" do
      #   weights["hidden_0"].should eq([0.0]) # TODO: for debugging; remove before merging to master
      # end

      # it "DEBUG hidden_1 values" do
      #   weights["hidden_1"].should eq([0.0]) # TODO: for debugging; remove before merging to master
      # end
    end


    # describe "#train" do
    #   it "TODO" do
    #     # ...
    #   end
    # end

    describe "#eval" do
      it "returns expected results" do
        net.init_hidden_nodes.should eq(default_expected_nodes_hidden)
      end
    end

    # describe "#backpropagate" do
    #   it "TODO" do
    #     # ...
    #   end
    # end

    # describe "#calculate_error" do
    #   it "TODO" do
    #     # ...
    #   end
    # end
  end
end



# icr
# require "./src/ai4cr/neural_network/rnnbim/net.cr"
# net = Ai4cr::NeuralNetwork::Rnnbim::Net.new
# puts net.pretty_inspect

# puts "net.input_state_range: #{net.input_state_range}"
# puts "net.hidden_state_range: #{net.hidden_state_range}"
# puts "net.output_state_range: #{net.output_state_range}"
# puts "net.nodes_out: #{net.nodes_out}"

# time_column_index = 0
# hidden_layer_index = 0
# w = net.init_weights_to_current_past(time_column_index, hidden_layer_index)

# w = net.init_weights_to_current_future(time_column_index, hidden_layer_index)

# w = net.init_network_weights

# puts "w: #{w.pretty_inspect}"
          
# /home/drhuffman/_dev_/github.com/drhuffman12/ai4cr_alt/src/ai4cr/neural_network/rnnbim/net.cr    

# mkdir spec/ai4cr/neural_network/rnnbim/net

# crystal spec spec/ai4cr/neural_network/rnnbim/net_spec.rb


