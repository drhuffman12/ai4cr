module Ai4cr
  module NeuralNetwork
    module Cmn
      class MiniNetManager < Breed::Manager(MiniNet)
        def mix_parts(child : MiniNet, parent_a : MiniNet, parent_b : MiniNet, delta)
          # re calc_guess:
          learning_rate = mix_one_part_number(parent_a.learning_rate, parent_b.learning_rate, delta)
          child.learning_rate = learning_rate

          momentum = mix_one_part_number(parent_a.momentum, parent_b.momentum, delta)
          child.momentum = momentum

          deriv_scale = mix_one_part_number(parent_a.deriv_scale, parent_b.deriv_scale, delta)
          child.deriv_scale = deriv_scale

          weights = mix_nested_parts(parent_a.weights, parent_b.weights, delta)
          child.weights = weights

          inputs_given = mix_nested_parts(parent_a.inputs_given, parent_b.inputs_given, delta)
          child.inputs_given = inputs_given

          outputs_guessed = mix_nested_parts(parent_a.outputs_guessed, parent_b.outputs_guessed, delta)
          child.outputs_guessed = outputs_guessed

          # re traing_and_adjust:
          outputs_expected = mix_nested_parts(parent_a.outputs_expected, parent_b.outputs_expected, delta)
          child.outputs_expected = outputs_expected

          output_deltas = mix_nested_parts(parent_a.output_deltas, parent_b.output_deltas, delta)
          child.output_deltas = output_deltas

          last_changes = mix_nested_parts(parent_a.last_changes, parent_b.last_changes, delta)
          child.last_changes = last_changes

          output_errors = mix_nested_parts(parent_a.output_errors, parent_b.output_errors, delta)
          child.output_errors = output_errors

          input_deltas = mix_nested_parts(parent_a.input_deltas, parent_b.input_deltas, delta)
          child.input_deltas = input_deltas

          # re error_stats:
          child.error_stats = Ai4cr::ErrorStats.new(parent_a.error_stats.history_size)

          child
        end
      end
    end
  end
end
