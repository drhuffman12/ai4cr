module Ai4cr
  module NeuralNetwork
    module Rnnbim # RNN, Bidirectional, Inversable Memory
      class Net # Ai4cr::NeuralNetwork::Rnnbim::Net
        alias IOStates = Array(Float64)
        alias ChronoIOStates = Array(Array(Float64))
        alias HiddenChannel = Hash(Symbol, ChronoIOStates)

        NODE_VAL_MIN = -1.0
        NODE_VAL_MID = 0.0
        NODE_VAL_MAX = 1.0

        getter time_column_qty, input_state_qty, output_state_qty
        getter hidden_layer_qty, hidden_layer_scale
        getter hidden_state_qty : Int32

        getter bias_enabled, bias_qty
        getter past_enabled, local_enabled, future_enabled

        getter time_column_range : Range(Int32, Int32)
        getter input_state_range : Range(Int32, Int32)
        getter output_state_range : Range(Int32, Int32)
        getter hidden_layer_range : Range(Int32, Int32)
        getter hidden_state_range : Range(Int32, Int32)
        
        getter hidden_channel_keys : Array(Symbol)

        getter hidden_delta_scales : Array(Int32)

        getter nodes_in : ChronoIOStates, nodes_out : ChronoIOStates
        getter nodes_hidden : Array(HiddenChannel) # , nodes_out

        # property nodes
        # property weights
        # property weights

        def initialize(
            @time_column_qty = 8, @input_state_qty = 4, @output_state_qty = 2,
            @hidden_layer_qty = 2, @hidden_layer_scale = 1.0,
            @bias_enabled = true, @past_enabled = true, @local_enabled = true, @future_enabled = true,
            # @local_offsets = [1,2,3,5,7,9,11] # TODO: e.g.: prime offsets
          )
          raise ArgumentError.new("The value of arg 'hidden_layer_qty' must be at least 1.") if hidden_layer_qty < 1

          @bias_qty = bias_enabled ? 1 : 0

          @time_column_range = (0..time_column_qty-1)
          @input_state_range = (0..input_state_qty-1)
          @output_state_range = (0..output_state_qty-1)

          @hidden_layer_range = (0..hidden_layer_qty-1)
          @hidden_state_qty = ((input_state_qty + output_state_qty) * hidden_layer_scale / 2.0).ceil.to_i32
          @hidden_state_range = (0..hidden_state_qty-1)

          @hidden_channel_keys = init_hidden_channel_keys

          @nodes_in = time_column_range.map { |t| input_state_range.map { |s| 0.0 } }
          @nodes_out = time_column_range.map { |t| output_state_range.map { |s| 0.0 } }
          @nodes_hidden = init_nodes_hidden

          @hidden_delta_scales = hidden_layer_range.map { |l| 2 ** l }
        end

        def init_hidden_channel_keys
          keys = [] of Symbol
          keys << :past if past_enabled
          keys << :local if local_enabled
          keys << :future if future_enabled
          keys << :combo
          keys
        end

        def init_nodes_hidden
          nh = [] of HiddenChannel
          hidden_layer_range.map do |layer|
            node_sets = HiddenChannel.new
            hidden_channel_keys.each do |key|
              node_sets[key] = time_column_range.map { |t| hidden_state_range.map { |s| 0.0 } }
            end
            nh << node_sets
          end
          nh
        end

        def init_nodes_channel
          time_column_range.map { |t| hidden_state_range.map { |s| 0.0 } }
        end


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
