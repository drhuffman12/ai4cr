require "./interface"
require "./../math"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module WeightSet
        module Memory
          include WeightSet::Interface
          
          property weights_memory : Array(WeightsSimple)

          def init_memory(time_column_index, output_node_set)
            output_node_set.memory_values_set.each do |memory_values|
              @weights_memory << output_node_set.state_values.map do |outs|
                w = memory_values.map do |inputs|
                  Ai4cr::NeuralNetwork::Rnn::Math.rnd_pos_neg_one
                end
                w
              end
            end
          end
        end
      end
    end
  end
end
