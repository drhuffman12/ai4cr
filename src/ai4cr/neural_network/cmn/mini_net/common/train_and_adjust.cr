require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNet
        module Common
          module TrainAndAdjust
            # ####
            # # TODO: Move prop and deriv methods to subclass and split method pairs per sub-class
            # def propagation_function
            #   ->(x : Float64) { x } # { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
            # end

            # # TODO: Move prop and deriv methods to subclass and split method pairs per sub-class
            # def derivative_propagation_function
            #   ->(y : Float64) { y } # { y*(1 - y) } # lambda { |y| 1.0 - y**2 }
            # end
            # ####
            abstract def propagation_function
            abstract def derivative_propagation_function

            # # training steps
            # TODO: utilize until_min_avg_error
            def train(inputs_given, outputs_expected, until_min_avg_error = 0.1)
              step_load_inputs(inputs_given)
              step_calc_forward
              # ...

              step_load_outputs(outputs_expected)
              step_calculate_error
              step_backpropagate

              # {outputs_guessed: @outputs_guessed, deltas: @deltas, error: @error}
              @error_total # @error
            end

            def step_load_outputs(outputs_expected)
              raise "Invalid outputs_expected size" if outputs_expected.size != @width
              load_outputs_expected(outputs_expected)
            end

            def step_calculate_error # aka calculate_error
              error = 0.0
              @outputs_expected.map_with_index do |oe, iw|
                error += 0.5*(oe - @outputs_guessed[iw])**2
              end
              @error_total = error
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
              @outputs_expected.map_with_index! { |v, i| outputs_expected[i] }
            end

            def load_outputs_deltas(outputs_deltas)
              @outputs_deltas.map_with_index! { |v, i| outputs_deltas[i] }
            end

            # Calculate deltas for output layer
            def step_calculate_output_deltas # (outputs_expected)
              @output_deltas.map_with_index! do |d, i|
                error = @outputs_expected[i] - @outputs_guessed[i]
                derivative_propagation_function.call(@outputs_guessed[i]) * error
              end
            end

            # Calculate deltas for hidden layers
            def step_calc_input_deltas # calculate_internal_deltas
              prev_deltas = @output_deltas
              layer_index = 1
              layer_deltas = [] of Float64
              height_considering_bias.times.to_a.each do |j|
                error = 0.0
                @width.times do |k|
                  error += @output_deltas[k] * @weights[j][k]
                end
                layer_deltas << (derivative_propagation_function.call(@inputs_given[j]) * error)
              end
              @input_deltas = layer_deltas
            end

            # Update weights after @deltas have been calculated.
            def step_update_weights # update_weights
              # per input row weights from first to last...
              # j == input row number
              height_considering_bias.times.to_a.each do |j|
                # per output column weights from first to last...
                # k == out column number
                @weights[j].each_with_index do |_elem, k|
                  change = @output_deltas[k]*@inputs_given[j]
                  @weights[j][k] += (learning_rate * change + momentum * @last_changes[j][k])
                  @last_changes[j][k] = change
                end
              end
            end

            # Calculate the radius of the error as if each output cell is an value in a coordinate set
            def step_calculate_error_distance_history
              # @error_distance_history_max = error_distance_history_max
              return @error_distance_history = [-1.0] if @error_distance_history_max < 1
              error = 0.0
              @outputs_expected.map_with_index do |oe, iw|
                error += (oe - @outputs_guessed[iw])**2
              end
              @error_distance = Math.sqrt(error)
              if @error_distance_history.size < @error_distance_history_max - 1
                # Array not 'full' yet, so add latest value to end
                @error_distance_history << error_total
              else
                # Array 'full', so rotate end to front and then put new value at last index
                @error_distance_history.rotate!
                @error_distance_history[-1] = error_total
              end
              @error_distance_history
            end
          end
        end
      end
    end
  end
end
