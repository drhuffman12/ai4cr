module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnSimpleConcerns
        module PropsAndInits
          # The 'io_offset' param is for setting, for a given time column, how much the inputs and outputs should be offset.
          #   For example, let's say the inputs and outputs are weather data and you want to guess tomorrow's weather based
          #     on today's and the past weather.
          #     * Setting 'io_offset' value to '-1' would mean that the outputs in tc # 0 would also represent weather data
          #       for day # -1 (which would be guessing yesterday's weather, which would overlap with the input data and
          #       probably not be of much help)
          #     * Setting 'io_offset' value to '0' would mean that the outputs in tc # 0 would also represent weather data
          #       for day # 0 (straight pass-thru; not good for guessing the future, but good for translating one set of data
          #       to another, like English to Spanish or speech to text)
          #     * Setting 'io_offset' value to '1' would mean that the outputs in tc # 0 would also represent weather data
          #       for day # 1 (and would let you guess tomorrow's weather based on today's and the past weather)
          IO_OFFSET_DEFAULT = 1

          TIME_COL_QTY_MIN      = 2
          HIDDEN_LAYER_QTY_MIN  = 1
          INPUT_SIZE_MIN        = 2
          OUTPUT_SIZE_MIN       = 1
          HIDDEN_SIZE_GIVEN_MIN = INPUT_SIZE_MIN + OUTPUT_SIZE_MIN

          getter io_offset : Int32
          getter time_col_qty : Int32

          getter input_size : Int32
          getter output_size : Int32
          getter hidden_layer_qty : Int32
          getter hidden_size_given : Int32?

          getter hidden_size : Int32

          property disable_bias : Bool
          property bias_default : Float64

          property learning_style : LearningStyle

          property learning_rate : Float64
          property momentum : Float64
          property deriv_scale : Float64

          getter synaptic_layer_qty : Int32

          # TODO: For 'errors', research using a key of an Enum instead of String. (Using Symbol's seems incompatible with 'from_json'.)
          getter errors : Hash(String, String)
          getter valid : Bool

          getter synaptic_layer_indexes : Array(Int32)
          getter time_col_indexes : Array(Int32)

          getter synaptic_layer_indexes_reversed : Array(Int32)
          getter time_col_indexes_reversed : Array(Int32)

          getter synaptic_layer_index_last : Int32
          getter time_col_index_last : Int32

          property node_output_sizes : Array(Int32)
          property node_input_sizes : Array(Array(NamedTuple(
            previous_synaptic_layer: Int32,
            previous_time_column: Int32)))

          property mini_net_set : Array(Array(Cmn::MiniNet))

          getter error_stats : Ai4cr::ErrorStats

          getter all_output_errors : Array(Array(Float64))

          getter input_set_given : Array(Array(Float64))
          getter output_set_expected : Array(Array(Float64))

          # TODO: Handle usage of a 'structure' param in 'initialize'
          # def initialize(@time_col_qty = TIME_COL_QTY_MIN, @structure = [INPUT_SIZE_MIN, OUTPUT_SIZE_MIN])
          #   initialize(time_col_qty, structure[0], structure[-1], structure[1..-2], )
          # end

          def config
            {
              io_offset:    @io_offset,
              time_col_qty: @time_col_qty,

              input_size:        @input_size,
              output_size:       @output_size,
              hidden_layer_qty:  @hidden_layer_qty,
              hidden_size_given: @hidden_size_given,

              disable_bias: @disable_bias,
              bias_default: @bias_default,

              learning_style: @learning_style,

              learning_rate: @learning_rate,
              momentum:      @momentum,
              deriv_scale:   @deriv_scale,
            }
          end

          def initialize(
            @io_offset = IO_OFFSET_DEFAULT,
            @time_col_qty = TIME_COL_QTY_MIN,

            @input_size = INPUT_SIZE_MIN,
            @output_size = OUTPUT_SIZE_MIN,
            @hidden_layer_qty = HIDDEN_LAYER_QTY_MIN,
            @hidden_size_given = nil,

            disable_bias : Bool? = nil,
            @bias_default = 1.0,

            @learning_style : LearningStyle = LS_RELU,

            learning_rate : Float64? = nil,
            momentum : Float64? = nil,
            @deriv_scale = rand / 2.0, # for Prelu

            history_size : Int32 = 10,
            name_suffix = ""
          )
            @name = init_name(name_suffix)

            # # init_network

            # TODO: Handle differing hidden layer output sizes
            if hidden_size_given.is_a?(Int32)
              @hidden_size = @hidden_size_given.as(Int32)
            else
              @hidden_size = @input_size + @output_size
            end

            # TODO: switch 'disabled_bias' to 'enabled_bias' and adjust defaulting accordingly
            @disable_bias = disable_bias.nil? ? false : disable_bias

            @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? rand : learning_rate.as(Float64)
            @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : rand

            @valid = false
            @errors = Hash(String, String).new
            validate!

            @synaptic_layer_qty = hidden_layer_qty + 1

            @synaptic_layer_indexes = calc_synaptic_layer_indexes
            @time_col_indexes = calc_time_col_indexes

            @synaptic_layer_indexes_reversed = @synaptic_layer_indexes.reverse
            @time_col_indexes_reversed = @time_col_indexes.reverse

            @synaptic_layer_index_last = @valid ? @synaptic_layer_indexes.last : -1
            @time_col_index_last = @valid ? @time_col_indexes.last : -1
            @node_output_sizes = calc_node_output_sizes
            @node_input_sizes = calc_node_input_sizes

            @mini_net_set = init_mini_net_set

            @error_stats = Ai4cr::ErrorStats.new(history_size)

            @all_output_errors = synaptic_layer_indexes.map { time_col_indexes.map { 0.0 } }

            @input_set_given = Array(Array(Float64)).new
            @output_set_expected = Array(Array(Float64)).new
          end

          def init_network
            # TODO: Handle differing hidden layer output sizes
            if hidden_size_given.is_a?(Int32)
              @hidden_size = @hidden_size_given.as(Int32)
            else
              @hidden_size = @input_size + @output_size
            end

            # TODO: switch 'disabled_bias' to 'enabled_bias' and adjust defaulting accordingly
            @disable_bias = disable_bias.nil? ? false : !!disable_bias

            @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? rand : learning_rate.as(Float64)
            @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : rand

            @valid = false
            @errors = Hash(String, String).new
            validate!

            @synaptic_layer_qty = hidden_layer_qty + 1

            @synaptic_layer_indexes = calc_synaptic_layer_indexes
            @time_col_indexes = calc_time_col_indexes

            @synaptic_layer_indexes_reversed = @synaptic_layer_indexes.reverse
            @time_col_indexes_reversed = @time_col_indexes.reverse

            @synaptic_layer_index_last = @valid ? @synaptic_layer_indexes.last : -1
            @time_col_index_last = @valid ? @time_col_indexes.last : -1
            @node_output_sizes = calc_node_output_sizes
            @node_input_sizes = calc_node_input_sizes

            @mini_net_set = init_mini_net_set

            @error_stats = Ai4cr::ErrorStats.new(history_size)

            @all_output_errors = synaptic_layer_indexes.map { time_col_indexes.map { 0.0 } }

            @input_set_given = Array(Array(Float64)).new
            @output_set_expected = Array(Array(Float64)).new
          end

          def valid?
            @valid
          end

          def validate!
            @errors = Hash(String, String).new

            @errors["time_col_qty"] = "time_col_qty must be at least #{TIME_COL_QTY_MIN}!" if time_col_qty < TIME_COL_QTY_MIN
            @errors["hidden_layer_qty"] = "hidden_layer_qty must be at least #{HIDDEN_LAYER_QTY_MIN}!" if hidden_layer_qty < HIDDEN_LAYER_QTY_MIN

            @errors["input_size"] = "input_size must be at least #{INPUT_SIZE_MIN}" if input_size < INPUT_SIZE_MIN
            @errors["output_size"] = "output_size must be at least #{OUTPUT_SIZE_MIN}" if output_size < OUTPUT_SIZE_MIN

            if hidden_size_given.is_a?(Int32)
              @errors["hidden_size_given"] = "hidden_size_given must be at least #{HIDDEN_SIZE_GIVEN_MIN} if supplied (otherwise it defaults to sum of @input_size and @output_size" if hidden_size_given.as(Int32) < HIDDEN_SIZE_GIVEN_MIN
            end

            @errors["io_offset"] = "io_offset must be a non-negative integer" if io_offset < 0

            @valid = errors.empty?
          end

          def calc_synaptic_layer_indexes
            if @valid
              Array.new(@synaptic_layer_qty) { |i| i }
            else
              [] of Int32
            end
          end

          def calc_time_col_indexes
            if @valid
              Array.new(@time_col_qty) { |i| i }
            else
              [] of Int32
            end
          end

          def calc_node_output_sizes
            if @valid
              synaptic_layer_indexes.map do |li|
                li == synaptic_layer_index_last ? output_size : hidden_size
              end
            else
              [] of Int32
            end
          end

          def calc_node_input_sizes
            if @valid
              input_sizes = [input_size] + node_output_sizes[0..-2]
              synaptic_layer_indexes.map do |li|
                in_size = input_sizes[li]
                output_size = node_output_sizes[li]
                time_col_indexes.map do |ti|
                  if ti == 0
                    {previous_synaptic_layer: in_size, previous_time_column: 0}
                  else
                    {previous_synaptic_layer: in_size, previous_time_column: output_size}
                  end
                end
              end
            else
              [[{previous_synaptic_layer: 0, previous_time_column: 0}]]
            end
          end

          def init_mini_net_set
            synaptic_layer_indexes.map do |li|
              # NOTE: It should suffice to have bias only on the first li nets.
              #   So, force bias only on 1st and none on others
              li_gt_0 = li != 0

              mn_output_size = node_output_sizes[li]
              time_col_indexes.map do |ti|
                mn_input_size = node_input_sizes[li][ti].values.sum
                Cmn::MiniNet.new(
                  height: mn_input_size,
                  width: mn_output_size,

                  learning_style: @learning_style,
                  deriv_scale: @deriv_scale,

                  disable_bias: li_gt_0,
                  bias_default: @bias_default,

                  learning_rate: @learning_rate,
                  momentum: @momentum,
                )
              end
            end
          end
        end
      end
    end
  end
end
