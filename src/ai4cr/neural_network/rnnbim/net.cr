module Ai4cr
  module NeuralNetwork
    module Rnnbim # RNN, Bidirectional, Inversable Memory
      class Net # Ai4cr::NeuralNetwork::Rnnbim::Net
        alias ChannelKey = Symbol # TODO: change to Enum?
        alias FromChannelKey = Symbol # TODO: change to Enum?
        alias ToChannelKey = Symbol # TODO: change to Enum?
        alias LayerName = String # TODO: change to Enum?

        alias NodesSimple = Array(Float64)
        alias NodesChrono = Array(NodesSimple)
        alias NodesChannel = Hash(ChannelKey, NodesChrono)
        alias NodesLayer = Hash(ChannelKey, NodesChannel)
        alias NodesHidden = Array(NodesLayer)

        alias WeightsSimple = Array(Array(Float64))
        alias WeightsFromChannel = Hash(FromChannelKey, WeightsSimple)
        alias WeightsToChannel = Hash(ToChannelKey,Array(WeightsFromChannel))
        alias WeightsNetwork = Hash(LayerName,WeightsToChannel)

        NODE_VAL_MIN = -1.0
        NODE_VAL_MID = 0.0
        NODE_VAL_MAX = 1.0

        getter time_column_scale, time_column_qty : Int32, input_state_qty, output_state_qty
        getter hidden_layer_qty, hidden_layer_scale
        getter hidden_state_qty : Int32

        getter time_column_range : Range(Int32, Int32)
        getter input_state_range : Range(Int32, Int32)
        getter output_state_range : Range(Int32, Int32)
        getter hidden_layer_range : Range(Int32, Int32)
        getter hidden_state_range : Range(Int32, Int32)

        getter hidden_channel_keys : Array(Symbol) # TODO: change to Array(Enum)?
        getter hidden_delta_scales : Array(Int32)

        getter nodes_in : NodesChrono, nodes_out : NodesChrono
        getter nodes_hidden : NodesHidden
        property network_weights : WeightsNetwork

        def initialize(
            @time_column_scale = 1, 
            @input_state_qty = 4, @output_state_qty = 2,
            @hidden_layer_qty = 2, @hidden_layer_scale = 1.0
            # ,
            # @local_offsets = [1,2,3,5,7,9,11] # TODO: e.g.: prime offsets
          )
          raise ArgumentError.new("The value of arg 'hidden_layer_qty' must be at least 1.") if hidden_layer_qty < 1

          @time_column_qty = 2 * (2 ** hidden_layer_qty) * time_column_scale
          @time_column_range = (0..time_column_qty-1)
          @input_state_range = (0..input_state_qty-1)
          @output_state_range = (0..output_state_qty-1)

          @hidden_layer_range = (0..hidden_layer_qty-1)
          @hidden_state_qty = ((input_state_qty + output_state_qty) * hidden_layer_scale / 2.0).ceil.to_i32
          @hidden_state_range = (0..hidden_state_qty-1)

          @hidden_channel_keys = [:past, :local, :future, :combo]

          @nodes_in = time_column_range.map { |t| input_state_range.map { |s| 0.0 } }
          @nodes_out = time_column_range.map { |t| output_state_range.map { |s| 0.0 } }
          @nodes_hidden = init_hidden_nodes

          @hidden_delta_scales = hidden_layer_range.map { |l| 2 ** l }

          @network_weights = init_network_weights
        end

        def init_hidden_nodes
          hn = NodesHidden.new
          hidden_layer_range.map do # |layer|
            node_sets = NodesChannel.new
            hidden_channel_keys.each do |channel_key|
              node_sets[channel_key] = time_column_range.map { |t| hidden_state_range.map { |s| 0.0 } }
            end
            hash = {
              :current => node_sets,
              :mem_same_image => node_sets.clone,
              :mem_after_image => node_sets.clone
            }
            hn << hash
          end
          hn
        end

        def init_network_weights
          
          nw = WeightsNetwork.new

          # output layer
          channel_key = :output
          node_sets = WeightsToChannel.new
          node_sets[channel_key] = time_column_range.map do |time_column_index|
            # hidden_state_range.map do |s|
              # init_weights_for_output_current(time_column_index)
              {
                :combo => [[0.0]],
                :bias => [[0.0]]
              }
            # end
          end
          nw["output"] = node_sets

          # hidden layers
          hidden_layer_range.each do |hidden_layer_index|
            # node_sets = HiddenChannelWeights.new
            node_sets = WeightsToChannel.new # Hash(Symbol,Array(WeightsFromChannel)).new
            hidden_channel_keys.each do |channel_key|
              node_sets[:past] = time_column_range.map { |time_column_index| init_weights_to_current_past(time_column_index, hidden_layer_index) }
              node_sets[:local] = time_column_range.map { |time_column_index| init_weights_to_current_local(time_column_index, hidden_layer_index) }
              node_sets[:future] = time_column_range.map { |time_column_index| init_weights_to_current_future(time_column_index, hidden_layer_index) }
              node_sets[:combo] = time_column_range.map { |time_column_index| init_weights_to_current_combo(time_column_index, hidden_layer_index) }
            end
            nw["hidden_#{hidden_layer_index}"] = node_sets
          end

          nw
        end

        def init_weights_to_current_past(time_column_index, hidden_layer_index)
          weights_from_channel = WeightsFromChannel.new

          if hidden_layer_index == 0
            weights_from_channel[:input] = init_weights_from_inputs_to_hidden
          else
            weights_from_channel[:combo] = init_weights_from_hidden_to_hidden
          end

          weights_from_channel[:past] = init_weights_from_hidden_to_hidden if time_column_index >= 2 ** hidden_layer_index
          weights_from_channel[:mem_same_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:mem_after_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          
          weights_from_channel
        end

        def init_weights_to_current_local(time_column_index, hidden_layer_index)
          weights_from_channel = WeightsFromChannel.new

          if hidden_layer_index == 0
            weights_from_channel[:input_past] = init_weights_from_inputs_to_hidden if time_column_index >= 2 ** hidden_layer_index
            weights_from_channel[:input_current] = init_weights_from_inputs_to_hidden
            weights_from_channel[:input_future] = init_weights_from_inputs_to_hidden if time_column_index < time_column_range.max - 2 ** hidden_layer_index
          else
            weights_from_channel[:combo_past] = init_weights_from_hidden_to_hidden if time_column_index >= 2 ** hidden_layer_index
            weights_from_channel[:combo_current] = init_weights_from_hidden_to_hidden
            weights_from_channel[:combo_future] = init_weights_from_hidden_to_hidden if time_column_index < time_column_range.max - 2 ** hidden_layer_index
          end

          weights_from_channel[:mem_same_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:mem_after_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          
          weights_from_channel
        end

        def init_weights_to_current_combo(time_column_index, hidden_layer_index)
          weights_from_channel = WeightsFromChannel.new

          if hidden_layer_index == 0
            weights_from_channel[:input] = init_weights_from_inputs_to_hidden
          else
            weights_from_channel[:combo] = init_weights_from_hidden_to_hidden
          end

          weights_from_channel[:past] = init_weights_from_hidden_to_hidden
          weights_from_channel[:local] = init_weights_from_hidden_to_hidden
          weights_from_channel[:future] = init_weights_from_hidden_to_hidden
          
          weights_from_channel[:mem_same_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:mem_after_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          
          weights_from_channel
        end

        def init_weights_to_current_future(time_column_index, hidden_layer_index)
          weights_from_channel = WeightsFromChannel.new

          weights_from_channel[:input] = init_weights_from_inputs_to_hidden if hidden_layer_index == 0
          weights_from_channel[:combo] = init_weights_from_hidden_to_hidden if hidden_layer_index > 0
          # weights_from_channel[:past] = init_weights_from_hidden_to_hidden if past_enabled && time_column_index > 0
          weights_from_channel[:future] = init_weights_from_hidden_to_hidden if time_column_index != time_column_range.max
          weights_from_channel[:mem_same_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:mem_after_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          weights_from_channel
        end

        def init_weights_to_current_output(time_column_index)
          weights_from_channel = WeightsFromChannel.new

          # weights_from_channel[:input] = init_weights_from_inputs_to_hidden if hidden_layer_index == 0
          weights_from_channel[:combo] = init_weights_from_hidden_to_hidden
          weights_from_channel[:past] = init_weights_from_hidden_to_hidden
          weights_from_channel[:mem_same_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:mem_after_image] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          weights_from_channel
        end

        def init_weights_from_inputs_to_hidden
          # time_column_range.map { |t| input_state_range.map { |i| hidden_state_range.map { |s| rnd_pos_neg_one } } }
          input_state_range.map { |i| hidden_state_range.map { |s| rnd_pos_neg_one } }
        end

        def init_weights_from_hidden_to_hidden
          hidden_state_range.map { |i| hidden_state_range.map { |s| rnd_pos_neg_one } }
        end

        def init_weights_from_input_to_output
          input_state_range.map { |i| output_state_range.map { |s| rnd_pos_neg_one } }
        end

        def init_weights_from_hidden_to_output
          output_state_range.map { |i| hidden_state_range.map { |s| rnd_pos_neg_one } }
        end

        def init_weights_from_bias_to_hidden
          [0].map { |i| hidden_state_range.map { |s| rnd_pos_neg_one } }
        end

        def rnd_pos_neg_one
          rand*2 - 1.0
        end
      end
    end
  end
end
