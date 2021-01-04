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

          def all_mini_net_inputs_given
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].inputs_given
              end
            end
          end

          def all_mini_net_input_deltas
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].input_deltas
              end
            end
          end

          def all_mini_net_last_changes
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].last_changes
              end
            end
          end

          def all_mini_net_momentum
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].momentum
              end
            end
          end

          def all_mini_net_output_deltas
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].output_deltas
              end
            end
          end

          def last_error_totals
            li = synaptic_layer_indexes.last
            time_col_indexes.map do |ti|
              mini_net_set[li][ti].error_total
            end
          end

          def error_total
            error = 0.0
            last_error_totals.each do |e|
              error += 0.5*(e)**2
            end
            error
          end
        end
      end
    end
  end
end
