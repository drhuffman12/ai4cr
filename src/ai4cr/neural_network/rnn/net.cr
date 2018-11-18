require "./../rnnbim/math"
# require "./../rnnbim/aliases"

module Ai4cr
  module NeuralNetwork
    module Rnn # RNN, Bidirectional, Inversable Memory
      class Net # Ai4cr::NeuralNetwork::Rnnbim::Net
        alias NodesSimple = Array(Float64)
        alias NodesChrono = Array(NodesSimple)
        alias NodesHiddenChrono = Array(NodesChrono) # RNN

        alias WeightsSimple = Array(Array(Float64))
        alias WeightsChrono = Array(WeightsSimple)
        alias WeightsHiddenChrono = Array(WeightsChrono) # RNN
        
        DENDRITE_OFFSETS_DEFAULT = [1] # [1,2,3,4,5,6,7,8] # [1,2,4,8,16,32,64,128]

        property hidden_layer_qty : Int32
        property time_column_qty : Int32
        property output_state_qty : Int32
        property input_state_qty : Int32

        property dendrite_offsets : Array(Int32)

        property hidden_layer_range : Range(Int32, Int32)
        property time_column_range : Range(Int32, Int32)
        property output_state_range : Range(Int32, Int32)
        property input_state_range : Range(Int32, Int32)

        property nodes_hidden_local : NodesHiddenChrono
        property nodes_hidden_past : NodesHiddenChrono
        property nodes_hidden_future : NodesHiddenChrono
        property nodes_hidden_combo : NodesHiddenChrono # The last one is for nodes_out
        # property nodes_out : NodesChrono

        property errors_hidden_local : NodesHiddenChrono
        property errors_hidden_past : NodesHiddenChrono
        property errors_hidden_future : NodesHiddenChrono
        property errors_hidden_combo : NodesHiddenChrono

        property deltas_hidden_local : NodesHiddenChrono
        property deltas_hidden_past : NodesHiddenChrono
        property deltas_hidden_future : NodesHiddenChrono
        property deltas_hidden_combo : NodesHiddenChrono

        property weights_hidden_local : WeightsHiddenChrono
        property weights_hidden_past : WeightsHiddenChrono
        property weights_hidden_future : WeightsHiddenChrono
        property weights_hidden_combo : WeightsHiddenChrono

        property nodes_in : NodesChrono
        property bias : Bool

        def initialize(@hidden_layer_qty = 1, @time_column_qty = 3, @output_state_qty = 2, @input_state_qty = 2, @dendrite_offsets = DENDRITE_OFFSETS_DEFAULT, @bias = true)
          @hidden_layer_range = (0..hidden_layer_qty-1)
          @time_column_range = (0..time_column_qty-1)
          @output_state_range = (0..output_state_qty-1)
          @input_state_range = (0..input_state_qty-(bias ? 0 : 1))

          @nodes_hidden_local = init_nodes_hidden_channel
          @nodes_hidden_past = init_nodes_hidden_channel
          @nodes_hidden_future = init_nodes_hidden_channel
          @nodes_hidden_combo = init_nodes_hidden_channel

          @errors_hidden_local = init_nodes_hidden_channel
          @errors_hidden_past = init_nodes_hidden_channel
          @errors_hidden_future = init_nodes_hidden_channel
          @errors_hidden_combo = init_nodes_hidden_channel

          @deltas_hidden_local = init_nodes_hidden_channel
          @deltas_hidden_past = init_nodes_hidden_channel
          @deltas_hidden_future = init_nodes_hidden_channel
          @deltas_hidden_combo = init_nodes_hidden_channel

          @weights_hidden_local = init_weights_hidden_channel
          @weights_hidden_past = init_weights_hidden_channel
          @weights_hidden_future = init_weights_hidden_channel
          @weights_hidden_combo = init_weights_hidden_channel

          @nodes_in = init_nodes_in
        end

        def init_nodes_hidden_channel
          hidden_layer_range.map { |l| time_column_range.map { |t| output_state_range.map { |s| 0.0 } } }
        end

        def init_weights_hidden_channel
          hidden_layer_range.map { |l| time_column_range.map { |t| output_state_range.map { |s| input_state_range.map { |s| Ai4cr::NeuralNetwork::Rnnbim::Math.rnd_pos_neg_one } } } }
        end

        def init_nodes_in
          time_column_range.map { |t| input_state_range.map { |s| 0.0 } }
        end

        def nodes_in(inputs : NodesChrono)
          @nodes_in = inputs.clone
        end
      end
    end
  end
end
