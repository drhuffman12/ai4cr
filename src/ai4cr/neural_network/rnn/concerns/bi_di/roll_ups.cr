module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module RollUps
            # def map_mini_nets(&block)
            #   synaptic_layer_indexes.map do |li|
            #     time_col_indexes.map do |ti|
            #       # mini_net_set[li][ti].outputs_guessed
            #       # block.call(mini_net_set[li][ti])
            #       yield mini_net_set[li][ti]
            #     end
            #   end
            # end

            def map_mini_nets(&block)
              empty_hash = Hash(Symbol, Array(Array(Float64))).new
              synaptic_layer_indexes.map do |sli|
                time_col_indexes.map do |tci|
                  channel = :channel_input_or_combo
                  w = {channel => block.call(mini_net_set[sli][tci][channel])}.merge(
                    if sli > 0
                      channel = :channel_backward
                      {channel => block.call(mini_net_set[sli][tci][channel])}
                    else
                      empty_hash
                    end
                  ).merge(
                    if sli > 0
                      channel = :channel_forward
                      {channel => block.call(mini_net_set[sli][tci][channel])}
                    else
                      empty_hash
                    end
                  )
                  w2 = Hash(Symbol, Array(Array(Float64))).new
                  w2[:channel_forward] = w[:channel_forward] if w.keys.includes?(:channel_forward)
                  w2[:channel_backward] = w[:channel_backward] if w.keys.includes?(:channel_backward)
                  w2[:channel_input_or_combo] = w[:channel_input_or_combo]

                  w2
                end
              end
            end

            def outputs
              all_mini_net_outputs
            end

            def all_mini_net_outputs
              all_mini_nets_map do |mini_net|
                mini_net.outputs_guessed
              end
            end

            def all_mini_net_weights
              # synaptic_layer_indexes.map do |li|
              #   time_col_indexes.map do |ti|
              #     mini_net_set[li][ti].weights
              #   end
              # end
              all_mini_nets_map do |mini_net|
                mini_net.weights
              end
            end

            def all_mini_net_inputs_given
              # synaptic_layer_indexes.map do |li|
              #   time_col_indexes.map do |ti|
              #     mini_net_set[li][ti].inputs_given
              #   end
              # end
              all_mini_nets_map do |mini_net|
                mini_net.inputs_given
              end
            end

            def all_mini_net_input_deltas
              # synaptic_layer_indexes.map do |li|
              #   time_col_indexes.map do |ti|
              #     mini_net_set[li][ti].input_deltas
              #   end
              # end
              all_mini_nets_map do |mini_net|
                mini_net.input_deltas
              end
            end

            def all_mini_net_last_changes
              # synaptic_layer_indexes.map do |li|
              #   time_col_indexes.map do |ti|
              #     mini_net_set[li][ti].last_changes
              #   end
              # end
              all_mini_nets_map do |mini_net|
                mini_net.last_changes
              end
            end

            def all_mini_net_momentum
              # synaptic_layer_indexes.map do |li|
              #   time_col_indexes.map do |ti|
              #     mini_net_set[li][ti].momentum
              #   end
              # end
              all_mini_nets_map do |mini_net|
                mini_net.momentum
              end
            end

            def all_mini_net_output_deltas
              # synaptic_layer_indexes.map do |li|
              #   time_col_indexes.map do |ti|
              #     mini_net_set[li][ti].output_deltas
              #   end
              # end
              all_mini_nets_map do |mini_net|
                mini_net.output_deltas
              end
            end

            def all_error_distances
              # synaptic_layer_indexes.map do |li|
              #   time_col_indexes.map do |ti|
              #     mini_net_set[li][ti].error_stats.distance
              #   end
              # end
              all_mini_nets_map do |mini_net|
                mini_net.error_stats.distance
              end
            end

            def all_error_distance_radius
              synaptic_layer_indexes.flat_map do |li|
                time_col_indexes.map do |ti|
                  # 0.5*(all_error_distances[li][ti])**2
                  all_error_distances[li][ti].values.map do |value|
                    0.5*(value)**2
                  end
                end
              end.sum
            end

            def all_output_errors
              # synaptic_layer_indexes.map do |li|
              #   time_col_indexes.map do |ti|
              #     mini_net_set[li][ti].output_errors
              #   end
              # end
              all_mini_nets_map do |mini_net|
                mini_net.error_stats.output_errors
              end
            end
          end
        end
      end
    end
  end
end
