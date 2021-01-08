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

          def all_error_totals
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].error_totals
              end
            end
          end

          def all_error_totals_radius
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                0.5*(all_error_totals[li][ti])**2
              end
            end.flatten.sum
          end

          def final_output_error_totals
            li = synaptic_layer_indexes.last
            time_col_indexes.map do |ti|
              mini_net_set[li][ti].error_total
            end
          end

          def final_output_error_totals_radius
            @error_total = final_output_error_totals.map { |e| 0.5*(e)**2 }.sum
          end

          def all_output_errors
            synaptic_layer_indexes.map do |li|
              time_col_indexes.map do |ti|
                mini_net_set[li][ti].output_errors
              end
            end
          end
        end
      end
    end
  end
end
