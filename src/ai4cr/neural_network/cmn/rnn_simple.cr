require "json"
require "./learning_style.cr"

# require "./rnn_concerns/*"

module Ai4cr
  module NeuralNetwork
    module Cmn
      # module ConnectedNetSet
      class RnnSimple # TODO!!!
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), and outputs

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
        IO_OFFSET_DEFAULT     = 1

        # NOTE: The first net should have a bias; the others should not.
        # TODO: Force bias only on 1st and none on others

        include JSON::Serializable

        TIME_COL_QTY_MIN      = 2
        HIDDEN_LAYER_QTY_MIN  = 1
        INPUT_SIZE_MIN        = 2
        OUTPUT_SIZE_MIN       = 1
        HIDDEN_SIZE_GIVEN_MIN = INPUT_SIZE_MIN + OUTPUT_SIZE_MIN

        getter hidden_layer_qty : Int32
        # getter nodal_layer_qty : Int32
        getter synaptic_layer_qty : Int32
        getter time_col_qty : Int32
        getter input_size : Int32
        getter output_size : Int32
        getter hidden_size : Int32
        getter hidden_size_given : Int32?
        getter io_offset : Int32

        getter errors : Hash(Symbol, String)
        getter valid : Bool

        # getter nodal_layer_indexes : Array(Int32)
        getter synaptic_layer_indexes : Array(Int32)
        getter time_col_indexes : Array(Int32)

        # getter nodal_layer_index_last : Int32
        getter synaptic_layer_index_last : Int32
        getter time_col_index_last : Int32

        property node_output_sizes : Array(Int32)
        property node_input_sizes : Array(Array(NamedTuple(previous_synaptic_layer: Int32, previous_time_column: Int32)))

        # TODO: Handle usage of a 'structure' param in 'initialize'
        # def initialize(@time_col_qty = TIME_COL_QTY_MIN, @structure = [INPUT_SIZE_MIN, OUTPUT_SIZE_MIN])
        #   initialize(time_col_qty, structure[0], structure[-1], structure[1..-2], )
        # end

        def initialize(
          @io_offset = IO_OFFSET_DEFAULT,
          @time_col_qty = TIME_COL_QTY_MIN,
          @input_size = INPUT_SIZE_MIN,
          @output_size = OUTPUT_SIZE_MIN,
          @hidden_layer_qty = HIDDEN_LAYER_QTY_MIN,
          @hidden_size_given = nil
        )
          @synaptic_layer_qty = hidden_layer_qty + 1
          # @nodal_layer_qty = 1 + synaptic_layer_qty

          # TODO: Handle differing hidden layer output sizes
          if hidden_size_given.is_a?(Int32)
            @hidden_size = @hidden_size_given.as(Int32)
          else
            @hidden_size = @input_size + @output_size
          end

          @valid = false
          @errors = Hash(Symbol, String).new
          validate!

          # @nodal_layer_indexes = calc_nodal_layer_indexes
          @synaptic_layer_indexes = calc_synaptic_layer_indexes
          @time_col_indexes = calc_time_col_indexes
          # @nodal_layer_index_last = @valid ? @nodal_layer_indexes.last : -1
          @synaptic_layer_index_last = @valid ? @synaptic_layer_indexes.last : -1
          @time_col_index_last = @valid ? @time_col_indexes.last : -1
          @node_output_sizes = calc_node_output_sizes
          @node_input_sizes = calc_node_input_sizes
        end

        def valid?
          @valid
        end

        def validate!
          @errors = Hash(Symbol, String).new

          @errors[:time_col_qty] = "time_col_qty must be at least #{TIME_COL_QTY_MIN}!" if time_col_qty < TIME_COL_QTY_MIN
          @errors[:hidden_layer_qty] = "hidden_layer_qty must be at least #{HIDDEN_LAYER_QTY_MIN}!" if hidden_layer_qty < HIDDEN_LAYER_QTY_MIN

          @errors[:input_size] = "input_size must be at least #{INPUT_SIZE_MIN}" if input_size < INPUT_SIZE_MIN
          @errors[:output_size] = "output_size must be at least #{OUTPUT_SIZE_MIN}" if output_size < OUTPUT_SIZE_MIN

          if hidden_size_given.is_a?(Int32)
            @errors[:hidden_size_given] = "hidden_size_given must be at least #{HIDDEN_SIZE_GIVEN_MIN} if supplied (otherwise it defaults to sum of @input_size and @output_size" if hidden_size_given.as(Int32) < HIDDEN_SIZE_GIVEN_MIN
          end

          @errors[:io_offset] = "io_offset must be a non-negative integer" if io_offset < 0

          @valid = errors.empty?
        end

        # def calc_nodal_layer_indexes
        #   if @valid
        #     Array.new(@nodal_layer_qty) { |i| i }
        #   else
        #     [] of Int32
        #   end
        # end

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
      end
    end
  end
end
