module Ai4cr
  module NeuralNetwork
    module Rnnbim # RNN, Bidirectional, Inversable Memory
      class Net # Ai4cr::NeuralNetwork::Rnnbim::Net
        alias IOStates = Array(Float64)
        alias ChronoIOStates = Array(Array(Float64))
        alias HiddenChannel = Hash(Symbol, ChronoIOStates)

        alias Weights = Array(Array(Float64))
        alias ChronoWeights = Array(Weights)
        alias HiddenChannelWeights = Hash(Symbol, ChronoWeights)
        alias HiddenChannelWeightMaps = Hash(Symbol, Hash(Symbol, Symbol))

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

        getter hidden_channel_keys : Array(Symbol)
        getter hidden_delta_scales : Array(Int32)

        getter nodes_in : ChronoIOStates, nodes_out : ChronoIOStates
        getter nodes_hidden : Array(Hash(Symbol, HiddenChannel))
        # Hash(Symbol, Hash(Symbol, Array(Array(Float64))))
        # Array(Hash(Symbol, Array(Array(Float64))))
        # Array(Hash(Symbol, HiddenChannel))

        # getter hidden_weights : Array(HiddenChannel)
        # getter output_weights : Array(HiddenChannel)

        def initialize(
          # @time_column_qty = 2 * 2 ** 2, 
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
          @nodes_hidden = init_nodes_hidden

          @hidden_delta_scales = hidden_layer_range.map { |l| 2 ** l }
        end

        def init_nodes_hidden
          nh = [] of Hash(Symbol, HiddenChannel)
          hidden_layer_range.map do |layer|
            node_sets = HiddenChannel.new
            hidden_channel_keys.each do |key|
              node_sets[key] = time_column_range.map { |t| hidden_state_range.map { |s| 0.0 } }
            end
            hash = {
              :current => node_sets,
              :mem_same_image => node_sets.clone,
              :mem_after_image => node_sets.clone
            }
            nh << hash
          end
          nh
        end

        def init_weights
          
          nh = Hash(String,Hash(Symbol,Array(Hash(Symbol, Array(Array(Float64)))))).new

          # output layer
          channel_key = :output
          node_sets = Hash(Symbol,Array(Hash(Symbol, Array(Array(Float64))))).new
          node_sets[channel_key] = time_column_range.map do |time_column_index|
            # hidden_state_range.map do |s|
              # init_weights_for_output_current(time_column_index)
              {
                :from_combo => [[0.0]],
                :from_bias => [[0.0]]
              }
            # end
          end
          nh["output"] = node_sets

          # hidden layers
          hidden_layer_range.each do |hidden_layer_index|
            # node_sets = HiddenChannelWeights.new
            node_sets = Hash(Symbol,Array(Hash(Symbol, Array(Array(Float64))))).new
            hidden_channel_keys.each do |channel_key|
              node_sets[:past] = time_column_range.map { |time_column_index| init_weights_to_current_past(time_column_index, hidden_layer_index) }
              node_sets[:local] = time_column_range.map { |time_column_index| init_weights_to_current_local(time_column_index, hidden_layer_index) }
              node_sets[:future] = time_column_range.map { |time_column_index| init_weights_to_current_future(time_column_index, hidden_layer_index) }
              node_sets[:combo] = time_column_range.map { |time_column_index| init_weights_to_current_combo(time_column_index, hidden_layer_index) }
            end
            nh["hidden_#{hidden_layer_index}"] = node_sets
          end

          nh
        end

        def init_weights_to_current_past(time_column_index, hidden_layer_index)
          weights = Hash(Symbol, Array(Array(Float64))).new

          if hidden_layer_index == 0
            weights[:from_inputs] = init_weights_from_inputs_to_hidden
          else
            weights[:from_combo] = init_weights_from_hidden_to_hidden
          end

          weights[:from_past] = init_weights_from_hidden_to_hidden if time_column_index >= 2 ** hidden_layer_index
          weights[:from_mem_same_image] = init_weights_from_hidden_to_hidden
          weights[:from_mem_after_image] = init_weights_from_hidden_to_hidden
          weights[:from_bias] = init_weights_from_bias_to_hidden
          
          weights
        end

        def init_weights_to_current_local(time_column_index, hidden_layer_index)
          weights = Hash(Symbol, Array(Array(Float64))).new

          if hidden_layer_index == 0
            weights[:from_inputs_past] = init_weights_from_inputs_to_hidden if time_column_index >= 2 ** hidden_layer_index
            weights[:from_inputs_current] = init_weights_from_inputs_to_hidden
            weights[:from_inputs_future] = init_weights_from_inputs_to_hidden if time_column_index < time_column_range.max - 2 ** hidden_layer_index
          else
            weights[:from_combo_past] = init_weights_from_hidden_to_hidden if time_column_index >= 2 ** hidden_layer_index
            weights[:from_combo_current] = init_weights_from_hidden_to_hidden
            weights[:from_combo_future] = init_weights_from_hidden_to_hidden if time_column_index < time_column_range.max - 2 ** hidden_layer_index
          end

          weights[:from_mem_same_image] = init_weights_from_hidden_to_hidden
          weights[:from_mem_after_image] = init_weights_from_hidden_to_hidden
          weights[:from_bias] = init_weights_from_bias_to_hidden
          
          weights
        end

        def init_weights_to_current_combo(time_column_index, hidden_layer_index)
          weights = Hash(Symbol, Array(Array(Float64))).new

          if hidden_layer_index == 0
            weights[:from_inputs] = init_weights_from_inputs_to_hidden
          else
            weights[:from_combo] = init_weights_from_hidden_to_hidden
          end

          weights[:from_past] = init_weights_from_hidden_to_hidden
          weights[:from_local] = init_weights_from_hidden_to_hidden
          weights[:from_future] = init_weights_from_hidden_to_hidden
          
          weights[:from_mem_same_image] = init_weights_from_hidden_to_hidden
          weights[:from_mem_after_image] = init_weights_from_hidden_to_hidden
          weights[:from_bias] = init_weights_from_bias_to_hidden
          
          weights
        end

        def init_weights_to_current_future(time_column_index, hidden_layer_index)
          weights = Hash(Symbol, Array(Array(Float64))).new

          weights[:from_inputs] = init_weights_from_inputs_to_hidden if hidden_layer_index == 0
          weights[:from_combo] = init_weights_from_hidden_to_hidden if hidden_layer_index > 0
          # weights[:from_past] = init_weights_from_hidden_to_hidden if past_enabled && time_column_index > 0
          weights[:from_future] = init_weights_from_hidden_to_hidden if time_column_index != time_column_range.max
          weights[:from_mem_same_image] = init_weights_from_hidden_to_hidden
          weights[:from_mem_after_image] = init_weights_from_hidden_to_hidden
          weights[:from_bias] = init_weights_from_bias_to_hidden
          weights
        end

        def init_weights_to_current_output(time_column_index)
          weights = Hash(Symbol, Array(Array(Float64))).new

          # weights[:from_inputs] = init_weights_from_inputs_to_hidden if hidden_layer_index == 0
          weights[:from_combo] = init_weights_from_hidden_to_hidden
          weights[:from_past] = init_weights_from_hidden_to_hidden
          weights[:from_mem_same_image] = init_weights_from_hidden_to_hidden
          weights[:from_mem_after_image] = init_weights_from_hidden_to_hidden
          weights[:from_bias] = init_weights_from_bias_to_hidden
          weights
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

        # def init_hidden_weights
        # end

        # def init_output_weights
        # end


        # @weights_to = {



        #   outputs: [] of Array(Float64),
        #   hidden: [
        #     {
        #       local_cur: [] of Array(Float64),
        #       hist_cur: [] of Array(Float64),
        #       fut_cur: [] of Array(Float64),
        #       combined_cur: [] of Array(Float64),

        #       local_mem: [] of Array(Float64),
        #       hist_mem: [] of Array(Float64),
        #       fut_mem: [] of Array(Float64),
        #       combined_mem: [] of Array(Float64)
        #     }
        #   ]
        # }
      end
    end
  end
end
