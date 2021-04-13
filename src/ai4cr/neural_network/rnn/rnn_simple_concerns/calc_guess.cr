module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnSimpleConcerns
        module CalcGuess
          # The 'io_offset' param is for setting, for a given time column, how much the inputs and outputs should be offset.
          #   For example, let's say the inputs and outputs are weather data and you want to guess tomorrow's weather based
          #     on today's and the past weather.
          #     * Setting 'io_offset' value to '-1' would mean that (we're just init'ing it or...) the outputs in tc
          #       number 0 would also represent weather data for day number -1 (which would be guessing yesterday's
          #       weather, which would overlap with the input data and probably not be of much help)
          #     * Setting 'io_offset' value to '0' would mean that the outputs in tc # 0 would also represent weather data
          #       for day # 0 (straight pass-thru; not good for guessing the future, but good for translating one set of data
          #       to another, like English to Spanish or speech to text)
          #     * Setting 'io_offset' value to '1' would mean that the outputs in tc # 0 would also represent weather data
          #       for day # 1 (and would let you guess tomorrow's weather based on today's and the past weather)
          getter io_offset = -1

          getter time_col_qty = -1

          getter input_size = -1
          getter output_size = -1
          getter hidden_layer_qty = -1
          getter hidden_size_given = 0

          getter hidden_size = -1

          property bias_disabled = false
          property bias_default : Float64 = 1.0

          property learning_styles = [LS_RELU, LS_SIGMOID]

          property learning_rate : Float64 = Ai4cr::Utils::Rand.rand_excluding
          property momentum : Float64 = Ai4cr::Utils::Rand.rand_excluding
          property deriv_scale : Float64 = Ai4cr::Utils::Rand.rand_excluding(scale: 0.5)

          getter synaptic_layer_qty : Int32

          # TODO: For 'errors', research using a key of an Enum instead of String. (Using Symbol's seems incompatible with 'from_json'.)
          getter errors = Hash(String, String).new
          getter valid = false

          getter synaptic_layer_qty = -1

          getter synaptic_layer_indexes = Array(Int32).new
          getter time_col_indexes = Array(Int32).new

          getter synaptic_layer_indexes_reversed = Array(Int32).new
          getter time_col_indexes_reversed = Array(Int32).new

          getter synaptic_layer_index_last = -1
          getter time_col_index_last = -1

          property node_output_sizes = Array(Int32).new
          property node_input_sizes = Array(Array(NamedTuple(
            previous_synaptic_layer: Int32,
            previous_time_column: Int32))).new

          property weight_init_scale : Float64 = 1.0
          property mini_net_set = Array(Array(Cmn::MiniNet)).new

          getter input_set_given = Array(Array(Float64)).new

          # steps for 'eval' aka 'guess':
          def eval(input_set_given)
            # TODO: Review/compare w/ 'train' and adjust as applicable!

            @input_set_given = input_set_given

            synaptic_layer_indexes.each do |li|
              time_col_indexes.each do |ti|
                mini_net_set[li][ti].step_load_inputs(inputs_for(li, ti))
                mini_net_set[li][ti].step_calc_forward
              end
            end

            outputs_guessed
          rescue ex
            msg = {
              my_msg:    "BROKE HERE!",
              file:      __FILE__,
              line:      __LINE__,
              klass:     ex.class,
              message:   ex.message,
              backtrace: ex.backtrace,
            }
            raise msg.to_s
          end

          def inputs_for(li, ti)
            case
            when li == 0 && ti == 0
              @input_set_given[ti]
            when li == 0 && ti > 0
              @input_set_given[ti] + step_outputs_guessed_from_previous_tc(li, ti)
            when li > 0 && ti == 0
              step_outputs_guessed_from_previous_li(li, ti)
            else
              step_outputs_guessed_from_previous_li(li, ti) + step_outputs_guessed_from_previous_tc(li, ti)
            end
          end

          def outputs_guessed
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              a = mini_net_set[li]
              b = a[ti]
              guessed = b.outputs_guessed
              guessed
            end
          end

          private def step_outputs_guessed_from_previous_tc(li, ti)
            raise "Index error in step_outputs_guessed_from_previous_tc" if ti == 0

            mini_net_set[li][ti - 1].outputs_guessed
          end

          private def step_outputs_guessed_from_previous_li(li, ti)
            raise "Index error in step_outputs_guessed_from_previous_li" if li == 0

            mini_net_set[li - 1][ti].outputs_guessed
          end

          # guesses
          def guesses_sorted
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].guesses_sorted
            end
          end

          def guesses_sorted
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].guesses_sorted
            end
          end

          def guesses_ceiled
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].guesses_ceiled
            end
          end

          def guesses_top_n(n)
            li = synaptic_layer_indexes.last

            time_col_indexes.map do |ti|
              mini_net_set[li][ti].guesses_top_n(n)
            end
          end
        end
      end
    end
  end
end
