# require "./aliases"

module Ai4cr
  module NeuralNetwork
    module Rnn
      # Math functions
      class Math
        # include Aliases
        # Math
        # def self.node_delta_scale(hidden_layer_index) # time_column_index
        #   2 ** hidden_layer_index
        # end
        
        # def self.node_scaled_border_past(hidden_layer_index)
        #   node_delta_scale(hidden_layer_index)
        # end
        
        # def self.node_scaled_border_future(time_column_range, hidden_layer_index)
        #   time_column_range.max - node_delta_scale(hidden_layer_index)
        # end

        def self.rnd_pos_neg_one
          rand*2 - 1.0
        end

        # def self.simple_weights_sum(ins : NodesSimple, simple_weights : WeightsSimple) : NodesSimple
        #   range_outs = (0..simple_weights.size-1) # TODO (?): use ranges from net
        #   range_ins = (0..simple_weights.first.size-1) # TODO (?): use ranges from net

        #   range_outs.map do |o|
        #     range_ins.map do |i|              
        #       ins[i] * simple_weights[o][i] 
        #     end.sum
        #   end
        # end

        # def self.simple_weights_sum_multi(ins_list : Array(NodesSimple), simple_weights_list : Array(WeightsSimple)) : Array(NodesSimple)
        #   ins_list.map_with_index do |ins, il|
        #     simple_weights = simple_weights_list[il]
        #     simple_weights_sum(ins, simple_weights)
        #   end
        # end

        # def self.simple_weights_sum_multi_sum(ins_list : Array(NodesSimple), simple_weights_list : Array(WeightsSimple)) : NodesSimple
        #   outs_list = simple_weights_sum_multi(ins_list, simple_weights_list)

        #   range_outs = (0..outs_list.size-1)
        #   range_ins_list = (0..outs_list.first.size-1)

        #   range_ins_list.map do |i|
        #     range_outs.reduce(0.0) { |n, o| n + outs_list[o][i] }
        #   end
        # end

        # # def self.simple_weights_propagate_0_to_1(ins_list : Array(NodesSimple), simple_weights_list : Array(WeightsSimple)) : NodesSimple
        # #   simple_weights_sum_multi_sum(ins_list, simple_weights_list).
        # #     map { |o| propagation_function_0_to_1.call(o) }
        # # end

        # def self.simple_weights_propagate_neg_1_to_1(ins_list : Array(NodesSimple), simple_weights_list : Array(WeightsSimple)) : NodesSimple
        #   simple_weights_sum_multi_sum(ins_list, simple_weights_list).
        #     map { |o| propagation_function_neg_1_to_1.call(o) }
        # end

        # # def self.simple_output_errors(expected_outs : NodesSimple, actual_outs : NodesSimple) : NodesSimple
        # #   # raise "index mis-match .. expected_outs.size (#{expected_outs.size}) != actual_outs.size (#{actual_outs.size})" if expected_outs.size != actual_outs.size
        # #   error_values = actual_outs.map_with_index do |ao, index|
        # #     expected_outs[index] - ao
        # #   end
        # #   # # puts "\n\nexpected_outs: #{expected_outs}, actual_outs: #{actual_outs}, error_values: #{error_values}, ao: #{ao}, index: #{index}\n\n"
        # #   # # puts "\n\nexpected_outs: #{expected_outs}"
        # #   # # puts "actual_outs: #{actual_outs}"
        # #   # # puts "error_values: #{error_values}\n\n"
        # #   error_values
        # # end

        # # def self.simple_output_deltas_0_to_1(expected_outs : NodesSimple, actual_outs : NodesSimple) : NodesSimple
        # #   error_values = simple_output_errors(expected_outs, actual_outs)
        # #   deltas = actual_outs.map_with_index do |ao, index|
        # #     error_values[index] * derivative_propagation_function_0_to_1.call(ao)
        # #   end
        # #   # # puts "\n\nexpected_outs: #{expected_outs}"
        # #   # # puts "actual_outs: #{actual_outs}"
        # #   # # puts "deltas: #{error_values}\n\n"
        # #   deltas
        # # end

        # # def self.simple_output_deltas_neg_1_to_1(expected_outs : NodesSimple, actual_outs : NodesSimple) : NodesSimple
        # #   error_values = simple_output_errors(expected_outs, actual_outs)
        # #   deltas = actual_outs.map_with_index do |ao, index|
        # #     error_values[index] * derivative_propagation_function_neg_1_to_1.call(ao)
        # #   end
        # #   # # puts "\n\nexpected_outs: #{expected_outs}"
        # #   # # puts "actual_outs: #{actual_outs}"
        # #   # # puts "deltas: #{error_values}\n\n"
        # #   deltas
        # # end

        # # # def self.simple_hidden_errors(expected_outs : NodesSimple, actual_outs : NodesSimple) : NodesSimple
        # # #   # raise "index mis-match .. expected_outs.size (#{expected_outs.size}) != actual_outs.size (#{actual_outs.size})" if expected_outs.size != actual_outs.size
        # # #   error_values = actual_outs.map_with_index do |ao, index|
        # # #     expected_outs[index] - ao
        # # #   end
        # # #   # # puts "\n\nexpected_outs: #{expected_outs}, actual_outs: #{actual_outs}, error_values: #{error_values}, ao: #{ao}, index: #{index}\n\n"
        # # #   # # puts "\n\nexpected_outs: #{expected_outs}"
        # # #   # # puts "actual_outs: #{actual_outs}"
        # # #   # # puts "error_values: #{error_values}\n\n"
        # # #   error_values
        # # # end

        # # def self.simple_hidden_delta_from_output_0_to_1(ins : NodesSimple, simple_weights : WeightsSimple, delta_outs : NodesSimple) : NodesSimple
        # #   range_outs = (0..simple_weights.first.size-1)
        # #   range_ins = (0..simple_weights.size-1)
        # #   range_ins.map do |i|  
        # #     range_outs.map do |o|
        # #       delta_outs[o] * simple_weights[i][o] 
        # #     end.sum * derivative_propagation_function_0_to_1.call(ins[i])
        # #   end
        # # end

        # # def self.simple_hidden_delta_from_output_neg_1_to_1(ins : NodesSimple, simple_weights : WeightsSimple, delta_outs : NodesSimple) : NodesSimple
        # #   range_outs = (0..simple_weights.first.size-1)
        # #   range_ins = (0..simple_weights.size-1)
        # #   range_ins.map do |i|  
        # #     range_outs.map do |o|
        # #       delta_outs[o] * simple_weights[i][o]
        # #     end.sum * derivative_propagation_function_neg_1_to_1.call(ins[i])
        # #   end
        # # end

        # # # simple_weight_changes aka last_changes
        # # def self.update_simple_weights(learning_rate : Float64, momentum : Float64, ins : NodesSimple, simple_weights : WeightsSimple, simple_weight_changes : WeightsSimple, delta_outs : NodesSimple) : NamedTuple(simple_weights: WeightsSimple, simple_weight_changes: WeightsSimple)
        # #   range_outs = (0..simple_weights.first.size-1)
        # #   range_ins = (0..simple_weights.size-1)
        # #   range_ins.each do |i|  
        # #     range_outs.map do |o|
        # #       # delta_outs[o] * simple_weights[i][o]
        # #       change = delta_outs[o]*ins[i]
        # #       simple_weights[i][o] += (learning_rate * change +
        # #                             momentum * simple_weight_changes[i][o])
        # #       simple_weight_changes[i][o] = change
        # #     end # .sum * derivative_propagation_function_0_to_1.call(ins[i])
        # #   end
        # #   {simple_weights: simple_weights, simple_weight_changes: simple_weight_changes}
        # # end

        # # # def self.simple_hidden_delta_from_hidden_neg_1_to_1(ins : NodesSimple, simple_weights : WeightsSimple, delta_outs : NodesSimple) : NodesSimple
        # # #   range_outs = (0..simple_weights.first.size-1)
        # # #   range_ins = (0..simple_weights.size-1)
        # # #   range_ins.map do |i|  
        # # #     range_outs.map do |o|
        # # #       delta_outs[o] * simple_weights[i][o]
        # # #     end.sum * derivative_propagation_function_neg_1_to_1.call(ins[i])
        # # #   end
        # # # end

        # # def self.propagation_function_0_to_1
        # #   ->(x : Float64) { 1/(1 + ::Math.exp(-1*(x))) }
        # # end

        # def self.propagation_function_neg_1_to_1
        #   ->(x : Float64) { ::Math.tanh(x) }
        # end

        # # def self.derivative_propagation_function_0_to_1
        # #   ->(y : Float64) { y*(1 - y) }
        # # end

        # # def self.derivative_propagation_function_neg_1_to_1
        # #   ->(y : Float64) { 1.0 - y**2 }
        # # end

      end
    end
  end
end
