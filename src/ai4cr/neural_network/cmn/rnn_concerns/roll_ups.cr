require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module RollUps
          def all_mini_net_outputs
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].outputs_guessed
              end
            end
          end

          def all_mini_net_weights
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].weights
              end
            end
          end
        end
      end
    end
  end
end
