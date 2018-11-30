require "./common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module WeightSet
        abstract class Hidden(CINS, SINS, ONS) < Common(CINS, SINS, ONS) # Ai4cr::NeuralNetwork::Rnn::WeightSet::Hidden
          def init_weights
            super

            output_node_set.memory_values_set.each do |memory_values|
              @weights << output_node_set.state_values.map do |outs|
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
