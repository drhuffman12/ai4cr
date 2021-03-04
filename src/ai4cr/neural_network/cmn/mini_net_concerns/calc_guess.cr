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
          # abstract def propagation_function

          property learning_style = LS_RELU

          property bias_default = 1.0
          property bias_disabled = false
          property learning_rate : Float64 = Ai4cr::Utils::Rand.rand_excluding
          property momentum : Float64 = Ai4cr::Utils::Rand.rand_excluding

          # for Prelu
          # TODO: set deriv_scale based on ?
          # @deriv_scale = 0.1,
          # @deriv_scale = 0.01,
          # @deriv_scale = 0.001,
          property deriv_scale : Float64 = Ai4cr::Utils::Rand.rand_excluding(scale: 0.5)

          getter width = -1
          getter height = -1

          getter height_considering_bias = -1
          getter width_indexes = Array(Int32).new
          getter height_indexes = Array(Int32).new

          property weights = Array(Array(Float64)).new

          property inputs_given = Array(Float64).new
          property outputs_guessed = Array(Float64).new

          def init_net_re_structure
            @height_considering_bias = @height + (@bias_disabled ? 0 : 1)
            @height_indexes = Array.new(@height_considering_bias) { |i| i }
            @width_indexes = Array.new(width) { |i| i }
            # Weight initialization (https://medium.com/datadriveninvestor/deep-learning-best-practices-activation-functions-weight-initialization-methods-part-1-c235ff976ed)
            # * Xavier initialization mostly used with tanh and logistic activation function
            # * He-initialization mostly used with ReLU or it’s variants — Leaky ReLU.
            @weights = @height_indexes.map { @width_indexes.map { Ai4cr::Utils::Rand.rand_neg_one_to_pos_one_no_zero } }
            # @weights = Array.new(height_considering_bias) { Array.new(width) { Ai4cr::Utils::Rand.rand_neg_one_to_pos_one_no_zero } }
            # @weights = @height_indexes.map { @width_indexes.map { Ai4cr::Utils::Rand.rand_neg_one_to_pos_one_no_zero*(Math.sqrt(2.0/(height_considering_bias + width))) } }
            # @weights = @height_indexes.map { @width_indexes.map { Ai4cr::Utils::Rand.rand_neg_one_to_pos_one_no_zero*(Math.sqrt(height_considering_bias/2.0)) } }
          end

          def init_net_re_guess
            @inputs_given = Array.new(@height_considering_bias, 0.0)
            @inputs_given[-1] = bias_default unless @bias_disabled
            @outputs_guessed = Array.new(width, 0.0)
          end

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
              x = propagation_function.call(sum)

              Ai4cr::Utils::Value.protect_against_extremes(x)
            end

            validate_outputs(@outputs_guessed, @width)
            validate_outputs(@outputs_guessed, @width_indexes.size)
          end

          def propagation_function
            # TODO: Make this JSON-loadable and customizable
            case @learning_style
            when LS_PRELU
              # LearningStyle::Prelu
              propagation_function_prelu
            when LS_RELU
              # LearningStyle::Relu
              propagation_function_relu
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

          def propagation_function_prelu
            # ->(x : Float64) { x < 0 ? 0.0 : x }
            # ->(x : Float64) {
            #   # For sake of limits on Math calculations, we'll max out the x value at 1000.
            #   return 0.0 if x < 0.0 || -x == Float64::NAN || -x == Float64::INFINITY
            #   return 1000.0 if x > 1000.0 || x == Float64::NAN || x == Float64::INFINITY
            #   x
            #   Ai4cr::Utils::Value.protect_against_extremes(x)
            # }
            ->(x : Float64) do
              v = x < 0 ? 0.0 : x
              Ai4cr::Utils::Value.protect_against_extremes(v)
            end
          end

          def propagation_function_relu
            # ->(x : Float64) { x < 0 ? 0.0 : [1.0, x].min }
            # ->(x : Float64) {
            #   return 0.0 if x < 0.0 || -x == Float64::NAN || -x == Float64::INFINITY
            #   return 1.0 if x > 1.0 || x == Float64::NAN || x == Float64::INFINITY
            #   # x
            #   Ai4cr::Utils::Value.protect_against_extremes(x)
            # }
            ->(x : Float64) do
              v = x < 0 ? 0.0 : [1.0, x].min
              Ai4cr::Utils::Value.protect_against_extremes(v)
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
