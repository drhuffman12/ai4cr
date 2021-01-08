require "json"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module MiniNetConcerns
        module CalcGuess
          # ####
          # # TODO: Move prop and deriv methods to subclass and split method pairs per sub-class
          # def propagation_function
          #   ->(x : Float64) { x } # { 1/(1 + Math.exp(-1*(x))) } # lambda { |x| Math.tanh(x) }
          # end
          abstract def propagation_function

          # steps for 'eval' aka 'guess':
          def eval(inputs_given) # aka eval
            step_load_inputs(inputs_given)
            step_calc_forward

            @outputs_guessed
          end

          def validate_inputs(inputs, height_expected)
            if (inputs.size > height_expected)
              raise "Invalid inputs given size: #{inputs.size}; should be no more than height_expected: #{height_expected}"
            end
          end

          def validate_outputs(outputs, width_expected)
            if outputs.size != width_expected
              raise "Invalid outputs size: #{outputs.size}; should be width_expected: #{width_expected}"
            end
          end

          def step_load_inputs(inputs_given)
            # Network could have a bias, which is tacked onto to the end of the inputs,
            # so we must account for that.
            validate_inputs(inputs_given, @height_considering_bias)

            load_inputs(inputs_given)
          end

          def load_inputs(inputs_given)
            # Avoid calling this directly; use 'step_load_inputs' instead.
            # Auto-convert non-float input values to float (using system default number of bits).
            # Also the inputs should NOT overwrite the bias slot, if any.
            inputs_given.each_with_index { |v, i| @inputs_given[i] = v.to_f }
          end

          def step_calc_forward # aka feedforward
            # TODO: Any removable dupe calls to these 'validate_*' methods?
            # TODO: Or, maybe move these to the 'validate!' method?

            validate_inputs(@inputs_given, @height_considering_bias)

            validate_outputs(@outputs_guessed, @width)
            validate_outputs(@outputs_guessed, @width_indexes.size)

            @outputs_guessed = @width_indexes.map do |w|
              sum = 0.0
              @height_indexes.each do |h|
                sum += @inputs_given[h]*@weights[h][w]
              end
              propagation_function.call(sum)
            end

            validate_outputs(@outputs_guessed, @width)
            validate_outputs(@outputs_guessed, @width_indexes.size)
          end

          def propagation_function
            # TODO: Make this JSON-loadable and customizable
            case @learning_style
            when LS_PRELU
              # LearningStyle::Prelu
              ->(x : Float64) { x < 0 ? 0.0 : x }
            when LS_RELU
              # LearningStyle::Rel
              ->(x : Float64) { x < 0 ? 0.0 : [1.0, x].min }
            when LS_SIGMOID
              # LearningStyle::Sigmoid
              ->(x : Float64) { 1/(1 + Math.exp(-1*(x))) }
            when LS_TANH
              # LearningStyle::Tanh
              ->(x : Float64) { Math.tanh(x) }
            else
              raise "Unsupported LearningStyle"
            end
          end

          # guesses
          def guesses_best
            # default set below, but might be different per subclass
            guesses_as_is
          end

          # outputs_guessed in sorted/top/bottom/etc order
          def guesses_as_is
            @outputs_guessed
          end

          def guesses_sorted
            @outputs_guessed.map_with_index { |o, idx| [idx, o].sort }
          end

          def guesses_rounded # good for MiniNet::Sigmoid; and maybe MiniNetRanh
            @outputs_guessed.map { |v| v.round }
          end

          def guesses_ceiled # good for MiniNetRelu
            @outputs_guessed.map { |v| v.ceil }
          end

          def guesses_top_n(n = @outputs_guessed.size)
            guesses_sorted[0..(n - 1)]
          end

          def guesses_bottom_n(n = @outputs_guessed.size)
            guesses_sorted.reverse[0..(n - 1)]
          end
        end
      end
    end
  end
end
