module Ai4cr
  module NeuralNetwork
    module Rnnbim # RNN, Bidirectional, Inversable Memory
      # Math functions
      class Math
        # Math
        def self.node_delta_scale(hidden_layer_index) # time_column_index
          2 ** hidden_layer_index
        end
        
        def self.node_scaled_border_past(hidden_layer_index)
          node_delta_scale(hidden_layer_index)
        end
        
        def self.node_scaled_border_future(time_column_range, hidden_layer_index)
          time_column_range.max - node_delta_scale(hidden_layer_index)
        end

        def self.rnd_pos_neg_one
          rand*2 - 1.0
        end

        def self.simple_weights_sum(ins : NodesSimple, simple_weights : WeightsSimple) : NodesSimple
          range_outs = (0..simple_weights.first.size-1)
          range_ins = (0..simple_weights.size-1)
          range_outs.map do |o|
            # range_ins.reduce(0.0) { |out, i| ins[i] * simple_weights[i][o] }
            range_ins.map { |i| ins[i] * simple_weights[i][o] }.sum
          end

        end

        def self.simple_weights_sum_multi(ins_list : Array(NodesSimple), simple_weights_list : Array(WeightsSimple)) : Array(NodesSimple)
          ins_list.map_with_index do |ins, il|
            # ins = ins_list[il]
            simple_weights = simple_weights_list[il]
            simple_weights_sum(ins, simple_weights)
          end
        end

        def self.simple_weights_sum_multi_sum(ins_list : Array(NodesSimple), simple_weights_list : Array(WeightsSimple)) : NodesSimple
          outs_list = simple_weights_sum_multi(ins_list, simple_weights_list)

          range_outs = (0..outs_list.first.size-1)
          range_ins_list = (0..outs_list.size-1)
          range_outs.map do |o|
            range_ins_list.reduce(0.0) { |out, i| outs_list[i][o] }
          end
        end

        def self.propagation_function_0_to_1
          ->(x : Float64) { 1/(1 + ::Math.exp(-1*(x))) }
        end

        def self.propagation_function_neg_1_to_1
          ->(x : Float64) { ::Math.tanh(x) }
        end

        def self.derivative_propagation_function_0_to_1
          ->(y : Float64) { y*(1 - y) }
        end

        def self.derivative_propagation_function_neg_1_to_1
          ->(y : Float64) { 1.0 - y**2 }
        end

      end
    end
  end
end
