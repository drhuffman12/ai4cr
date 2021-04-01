module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNetConcerns
        module TrainAndAdjust
          property outputs_expected = Array(Float64).new
          property output_deltas = Array(Float64).new
          property last_changes = Array(Array(Float64)).new # aka previous weights
          property output_errors = Array(Float64).new
          property input_deltas = Array(Float64).new

          def init_net_re_train
            @outputs_expected = Array.new(width, 0.0)
            @output_deltas = Array.new(width, 0.0)

            @last_changes = Array.new(@height_considering_bias, Array.new(width, 0.0))
            @output_errors = @width_indexes.map { 0.0 }
            @input_deltas = Array.new(@height_considering_bias, 0.0)
          end

          # # training steps
          # TODO: utilize until_min_avg_error
          def train(inputs_given, outputs_expected, until_min_avg_error = UNTIL_MIN_AVG_ERROR_DEFAULT)
            step_load_inputs(inputs_given)
            step_calc_forward

            step_load_outputs(outputs_expected)
            step_calc_output_errors
            step_backpropagate

            calculate_error_distance
          end

          def step_load_outputs(outputs_expected)
            raise "Invalid outputs_expected size" if outputs_expected.size != @width
            load_outputs_expected(outputs_expected)
          end

          def calculate_error_distance
            @error_stats.distance = @output_errors.sum { |e| 0.5 * e ** 2 }
          end

          def step_backpropagate
            step_calculate_output_deltas

            step_calc_input_deltas
            step_update_weights
            # auto_shrink_weights # for Relu; keep?
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
            @outputs_expected.map_with_index! { |_, i| outputs_expected[i].to_f }
          end

          def load_outputs_deltas(outputs_deltas)
            @outputs_deltas.map_with_index! { |_, i| outputs_deltas[i] }
          end

          # Calculate deltas for output layer
          def step_calculate_output_deltas # (outputs_expected)
            @output_deltas.map_with_index! do |_, i|
              derivative_propagation_function.call(@outputs_guessed[i].clone) * @output_errors[i].clone
            end
          end

          def step_calc_output_errors
            @output_errors = @outputs_guessed.map_with_index do |og, i|
              @outputs_expected[i] - og
            end
          end

          # Calculate deltas for hidden layers
          def step_calc_input_deltas # aka calculate_internal_deltas
            # NOTE: This takes into account the specified 'bias' value (where applicable)
            layer_deltas = [] of Float64
            height_indexes.each do |j|
              error = 0.0
              width_indexes.each do |k|
                error += @output_deltas[k] * @weights[j][k]
              end
              layer_deltas << (derivative_propagation_function.call(@inputs_given[j]) * error)
            end
            @input_deltas = layer_deltas
          end

          # Update weights after @deltas have been calculated.
          def step_update_weights
            # NOTE: This takes into account the specified 'bias' value (where applicable)
            step_update_weights_v1
            # step_update_weights_v2 # for larger weight sets, this become too much overhead
          end

          # def auto_shrink_weights
          #   # How best to handle auto-scaling down?
          #   if @output_deltas.map{|d| d.abs}.sum > @outputs_guessed.size
          #     height_indexes.each do |j|
          #       @weights[j].each_with_index do |_elem, k|
          #         # change = @output_deltas[k]*@inputs_given[j]
          #         # weight_delta = (@learning_rate * change + @momentum * @last_changes[j][k])
          #         # @weights[j][k] += weight_delta
          #         # @last_changes[j][k] = change
          #         @weights[j][k] = @weights[j][k] / (2*@outputs_guessed.size)
          #       end
          #     end
          #   end
          # end

          # Update weights after @deltas have been calculated.
          def step_update_weights_v1
            height_indexes.each do |j|
              @weights[j].each_with_index do |_elem, k|
                change = @output_deltas[k]*@inputs_given[j]
                weight_delta = (@learning_rate * change + @momentum * @last_changes[j][k])
                @weights[j][k] += weight_delta
                @last_changes[j][k] = change
              end
            end
          end

          # Update weights after @deltas have been calculated.
          def step_update_weights_v2
            channel = Channel(Nil).new

            height_indexes.each do |j|
              @weights[j].each_with_index do |_elem, k|
                spawn do
                  change = @output_deltas[k]*@inputs_given[j]
                  @weights[j][k] += (@learning_rate * change + @momentum * @last_changes[j][k])
                  @last_changes[j][k] = change
                  channel.send(nil)
                end
              end
            end

            height_indexes.each do |j|
              @weights[j].each do
                channel.receive
              end
            end
          end

          # Per Learning Style:
          def set_deriv_scale_prelu(scale)
            @deriv_scale = scale
          end

          def derivative_propagation_function
            # TODO: Make this JSON-loadable and customizable
            case @learning_style
            when LS_PRELU
              # LearningStyle::Prelu
              ->(y : Float64) { y < 0 ? @deriv_scale : 1.0 }
            when LS_RELU
              # LearningStyle::Rel
              ->(y : Float64) { y < 0 ? 0.0 : 1.0 }
            when LS_SIGMOID
              # LearningStyle::Sigmoid
              ->(y : Float64) { y*(1 - y) }
            when LS_TANH
              # LearningStyle::Tanh
              ->(y : Float64) { 1.0 - (y**2) }
            else
              raise "Unsupported LearningStyle"
            end
          end

          def guesses_best
            # TODO: Make this JSON-loadable and customizable
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
