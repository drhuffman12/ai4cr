module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module Common
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

            def all_error_distances
              synaptic_layer_indexes.map do |li|
                time_col_indexes.map do |ti|
                  mini_net_set[li][ti].error_stats.distance
                end
              end
            end

            def all_error_distance_radius
              synaptic_layer_indexes.flat_map do |li|
                time_col_indexes.map do |ti|
                  0.5*(all_error_distances[li][ti])**2
                end
              end.sum
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
end
