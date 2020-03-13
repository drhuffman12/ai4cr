require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNetConcerns
        module TrainAndAdjust
          # abstract def derivative_propagation_function

          # # training steps
          # TODO: utilize until_min_avg_error
          def train(inputs_given, outputs_expected, until_min_avg_error = 0.1)
            step_load_inputs(inputs_given)
            step_calc_forward

            step_load_outputs(outputs_expected)
            step_calculate_error

            step_backpropagate

            @error_total
          end

          def step_load_outputs(outputs_expected)
            raise "Invalid outputs_expected size; outputs_expected.size: #{outputs_expected.size}, width: #{@width}" if outputs_expected.size != @width
            load_outputs_expected(outputs_expected)
          end

          def step_calculate_error
            error = 0.0
            @outputs_expected.map_with_index do |oe, iw|
              error += 0.5*(oe - @outputs_guessed[iw])**2
            end
            @error_total = error
            @error_total
          end

          def step_backpropagate
            step_calculate_output_deltas

            step_calc_input_deltas
            step_update_weights
          end

          # This would be a chained MiniNet's input_deltas
          # e.g.: mini_net_A feeds is chained into mini_net_B
          #    So you would mini_net_A.step_load_chained_outputs_deltas(mini_net_B.input_deltas)
          def step_load_chained_outputs_deltas(outputs_deltas)
            raise "Invalid outputs_deltas size" if outputs_expected.size != @width
            load_outputs_deltas(outputs_deltas)
          end

          # private

          def load_outputs_expected(outputs_expected)
            @outputs_expected.map_with_index! { |_, i| outputs_expected[i] }
          end

          def load_outputs_deltas(outputs_deltas)
            @outputs_deltas.map_with_index! { |_, i| outputs_deltas[i] }
          end

          # Calculate deltas for output layer
          def step_calculate_output_deltas # (outputs_expected)
            @output_deltas.map_with_index! do |_, i|
              error = @outputs_expected[i] - @outputs_guessed[i]
              derivative_propagation_function.call(@outputs_guessed[i]) * error
              # # TODO: Research ReLU and why I'm not seeing performance gain in my code
              # # For Relu performance gain, check for 0.0
              # der_val = derivative_propagation_function.call(@outputs_guessed[i])
              # der_val == 0.0 ? 0.0 : der_val * error
            end
          end

          # Calculate deltas for hidden layers
          def step_calc_input_deltas # aka calculate_internal_deltas
            # prev_deltas = @output_deltas
            # layer_index = 1
            layer_deltas = [] of Float64
            height_indexes.each do |j|
              error = 0.0
              width_indexes.each do |k|
                error += @output_deltas[k] * @weights[j][k]
              end
              layer_deltas << (derivative_propagation_function.call(@inputs_given[j]) * error)
              # # TODO: Research ReLU and why I'm not seeing performance gain in my code
              # # For Relu performance gain, check for 0.0
              # der_val = derivative_propagation_function.call(@inputs_given[j])
              # layer_deltas << (der_val == 0.0 ? 0.0 : der_val * error)
            end
            @input_deltas = layer_deltas
          end

          # Update weights after @deltas have been calculated.
          def step_update_weights
            height_indexes.each do |j|
              @weights[j].each_with_index do |_elem, k|
                change = @output_deltas[k]*@inputs_given[j]
                @weights[j][k] += (@learning_rate * change + @momentum * @last_changes[j][k])
                @last_changes[j][k] = change
              end
            end
          end

          # Calculate the radius of the error as if each output cell is an value in a coordinate set
          def step_calculate_error_distance_history
            return @error_distance_history = [-1.0] if @error_distance_history_max < 1
            if @error_distance_history.size < @error_distance_history_max - 1
              # Array not 'full' yet, so add latest value to end
              @error_distance_history << @error_total
            else
              # Array 'full', so rotate end to front and then put new value at last index
              @error_distance_history.rotate!
              @error_distance_history[-1] = @error_total
            end
            @error_distance_history
          end

          # Per Learning Style:
          def set_deriv_scale_prelu(scale)
            @deriv_scale = scale
          end

          def propagation_function
            case @learning_style
            when LS_PRELU # LearningStyle::Prelu
              ->(x : Float64) { x < 0 ? 0.0 : [1.0, x].min }
            when LS_RELU # LearningStyle::Rel
              ->(x : Float64) { x < 0 ? 0.0 : [1.0, x].min }
            when LS_SIGMOID                                # LearningStyle::Sigmoid
              ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
            when LS_TANH                                   # LearningStyle::Tanh
              ->(x : Float64) { Math.tanh(x) }
            else
              raise "Unsupported LearningStyle"
            end
          end

          def derivative_propagation_function
            case @learning_style
            when LS_PRELU # LearningStyle::Prelu
              ->(y : Float64) { y < 0 ? @deriv_scale : 1.0 }
            when LS_RELU # LearningStyle::Rel
              ->(y : Float64) { y < 0 ? 0.0 : 1.0 }
            when LS_SIGMOID                 # LearningStyle::Sigmoid
              ->(y : Float64) { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
            when LS_TANH                    # LearningStyle::Tanh
              ->(y : Float64) { 1.0 - (y**2) }
            else
              raise "Unsupported LearningStyle"
            end
          end

          def guesses_best
            case @learning_style
            when LS_PRELU # LearningStyle::Prelu
              guesses_ceiled
            when LS_RELU # LearningStyle::Rel
              guesses_ceiled
            when LS_SIGMOID # LearningStyle::Sigmoid
              guesses_rounded
            when LS_TANH # LearningStyle::Tanh
              guesses_rounded
            else
              raise "Unsupported LearningStyle"
            end
          end
        end
      end
    end
  end
end
