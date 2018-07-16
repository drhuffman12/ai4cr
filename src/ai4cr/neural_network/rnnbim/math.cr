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
            # range_ins.map { |i| ins[i] * simple_weights[i][o] }.sum
            range_ins.map do |i|              
              # puts "\n\nsimple_weights_sum(ins: #{ins}, simple_weights: #{simple_weights}, range_outs: #{range_outs}, range_ins: #{range_ins}, o: #{o}, i: #{i})\n\n"
              ins[i] * simple_weights[i][o] 
            end.sum
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
            range_ins_list.reduce(0.0) { |n, i| outs_list[i][o] }
          end
        end

        def self.simple_weights_propagate(ins_list : Array(NodesSimple), simple_weights_list : Array(WeightsSimple)) : NodesSimple
          simple_weights_sum_multi_sum(ins_list, simple_weights_list).
            map { |o| propagation_function_neg_1_to_1.call(o) }
        end

        def self.simple_output_errors(expected_outs : NodesSimple, actual_outs : NodesSimple) : NodesSimple
          # raise "index mis-match .. expected_outs.size (#{expected_outs.size}) != actual_outs.size (#{actual_outs.size})" if expected_outs.size != actual_outs.size
          error_values = actual_outs.map_with_index do |ao, index|
            expected_outs[index] - ao
          end
          # puts "\n\nexpected_outs: #{expected_outs}, actual_outs: #{actual_outs}, error_values: #{error_values}, ao: #{ao}, index: #{index}\n\n"
          # puts "\n\nexpected_outs: #{expected_outs}"
          # puts "actual_outs: #{actual_outs}"
          # puts "error_values: #{error_values}\n\n"
          error_values
        end

        def self.simple_output_deltas(expected_outs : NodesSimple, actual_outs : NodesSimple) : NodesSimple
          error_values = simple_output_errors(expected_outs, actual_outs)
          actual_outs.map_with_index do |ao, index|
            # error = expected_outs[index] - ao
            # scale = derivative_propagation_function_neg_1_to_1.call(ao)
            # error * scale
            # error_values[index] * scale
            error_values[index] * derivative_propagation_function_neg_1_to_1.call(ao)
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
