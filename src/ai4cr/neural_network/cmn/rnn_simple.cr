require "json"
require "./learning_style.cr"

# require "./rnn_concerns/*"

module Ai4cr
  module NeuralNetwork
    module Cmn
      # module ConnectedNetSet
      class RnnSimple # TODO!!!
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), and outputs

        # NOTE: The first net should have a bias; the others should not.
        # TODO: Force bias only on 1st and none on others

        include JSON::Serializable

        TIME_COL_QTY_MIN      = 2
        HIDDEN_LAYER_QTY_MIN  = 1
        INPUT_SIZE_MIN        = 2
        OUTPUT_SIZE_MIN       = 1
        HIDDEN_SIZE_GIVEN_MIN = INPUT_SIZE_MIN + OUTPUT_SIZE_MIN

        getter hidden_layer_qty : Int32
        getter layer_qty : Int32
        getter time_col_qty : Int32
        getter input_size : Int32
        getter output_size : Int32
        getter hidden_size : Int32
        getter hidden_size_given : Int32?

        getter errors : Hash(Symbol, String)
        getter valid : Bool

        getter layer_indexes : Array(Int32)
        getter time_col_indexes : Array(Int32)

        getter layer_index_last : Int32
        getter time_col_index_last : Int32

        property node_output_sizes : Array(Int32)
        property node_input_sizes : Array(Array(NamedTuple(current_tc: Int32, prev_tc: Int32)))

        def initialize(
          @hidden_layer_qty = HIDDEN_LAYER_QTY_MIN,
          @time_col_qty = TIME_COL_QTY_MIN,
          @input_size = INPUT_SIZE_MIN,
          @output_size = OUTPUT_SIZE_MIN,
          @hidden_size_given = nil
        )
          @layer_qty = 1 + hidden_layer_qty + 1 # in, hiddens, out
          if hidden_size_given.is_a?(Int32)
            # unless hidden_size_given.nil?
            @hidden_size = @hidden_size_given.as(Int32)
          else
            @hidden_size = @input_size + @output_size
          end

          @valid = false
          @errors = Hash(Symbol, String).new
          validate!

          @layer_indexes = calc_layer_indexes
          @time_col_indexes = calc_time_col_indexes
          @layer_index_last = @valid ? @layer_indexes.last : -1
          @time_col_index_last = @valid ? @time_col_indexes.last : -1
          @node_output_sizes = calc_node_output_sizes
          @node_input_sizes = calc_node_input_sizes

          # pin = pre_init_network
          # @time_col_indexes = pin[:time_col_indexes]
          # @layer_indexes = pin[:layer_indexes]
          # @layer_index_last = pin[:layer_index_last]
          # @time_col_index_last = pin[:time_col_index_last]
          # @node_input_sizes = pin[:node_input_sizes]
          # @node_output_sizes = pin[:node_output_sizes]

          # init_network if valid?
          # raise "INVALID" unless @valid
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
            # unless hidden_size_given.nil?
            @errors[:hidden_size_given] = "hidden_size_given must be at least #{HIDDEN_SIZE_GIVEN_MIN} if supplied (otherwise it defaults to sum of @input_size and @output_size" if hidden_size_given.as(Int32) < HIDDEN_SIZE_GIVEN_MIN
          end

          @valid = errors.empty?
        end

        def calc_layer_indexes
          if @valid
            Array.new(@layer_qty) { |i| i }
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

        # def pre_init_network
        #   if @valid
        #     puts "pre_init_network .. valid"
        #     @layer_indexes = Array.new(@layer_qty) { |i| i }
        #     @time_col_indexes = Array.new(@time_col_qty) { |i| i }

        #     @layer_index_last = @layer_indexes.last
        #     @time_col_index_last = @time_col_indexes.last

        #     # node_output_sizes # TODO: move to method
        #     @node_output_sizes = calc_node_output_sizes
        #     @node_input_sizes = calc_node_input_sizes
        #   else
        #     puts "pre_init_network .. invalid"
        #     @time_col_indexes = [] of Int32
        #     @layer_indexes = [] of Int32
        #     @layer_index_last = 0
        #     @time_col_index_last = 0
        #     @node_input_sizes = [[{current_tc: 0, prev_tc: 0}]]
        #     # @node_input_sizes = [[NamedTuple(current_tc: Int32, prev_tc: Int32).new]]
        #     @node_output_sizes = [] of Int32
        #   end

        #   {
        #     time_col_indexes: @time_col_indexes,
        #     layer_indexes: @layer_indexes,
        #     layer_index_last: @layer_index_last,
        #     time_col_index_last: @time_col_index_last,
        #     node_input_sizes: @node_input_sizes,
        #     time_col_indnode_output_sizesexes: @node_output_sizes,
        #   }
        # end

        def calc_node_output_sizes
          if @valid
            layer_indexes.map do |li|
              li == layer_index_last ? output_size : hidden_size
            end
          else
            [] of Int32
          end
        end

        def calc_node_input_sizes
          if @valid
            input_sizes = [input_size] + node_output_sizes[0..-2]
            layer_indexes.map do |li|
              in_size = input_sizes[li]
              output_size = node_output_sizes[li]
              time_col_indexes.map do |ti|
                if ti == 0
                  {current_tc: in_size, prev_tc: 0}
                  # {current_tc: node_output_sizes[li], prev_tc: 0}
                else
                  # {current_tc: input_size, prev_tc: output_size}
                  {current_tc: in_size, prev_tc: output_size}
                end
              end
            end
          else
            [[{current_tc: 0, prev_tc: 0}]]
          end
        end
      end
    end
  end
end
