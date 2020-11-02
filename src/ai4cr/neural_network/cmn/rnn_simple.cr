require "json"
require "./learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      # module ConnectedNetSet
      class RnnSimple # TODO!!!
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), and outputs

        # NOTE: The first net should have a bias; the others should not.
        # TODO: Force bias only on 1st and none on others

        include JSON::Serializable

        TIME_COL_QTY_MIN = 2
        HIDDEN_LAYER_QTY_MIN = 1
        INPUT_SIZE_MIN = 2
        OUTPUT_SIZE_MIN = 1
        HIDDEN_SIZE_GIVEN_MIN = INPUT_SIZE_MIN + OUTPUT_SIZE_MIN


        getter time_col_qty : Int32
        getter hidden_layer_qty : Int32
        getter input_size : Int32
        getter output_size : Int32
        getter hidden_size : Int32
        getter hidden_size_given : Int32?
        
        getter errors : Hash(Symbol, String)
        getter valid : Bool

        def initialize(
          @time_col_qty = TIME_COL_QTY_MIN,
          @hidden_layer_qty = HIDDEN_LAYER_QTY_MIN,
          @input_size = INPUT_SIZE_MIN,
          @output_size = OUTPUT_SIZE_MIN,
          @hidden_size_given = nil
        )
          if hidden_size_given.is_a?(Int32)
          # unless hidden_size_given.nil?
            @hidden_size = @hidden_size_given.as(Int32)
          else
            @hidden_size = @input_size + @output_size
          end

          @valid = false
          @errors = Hash(Symbol, String).new
          validate

          # raise "INVALID" unless @valid
        end

        def valid?
          @valid
        end

        def validate
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

        # getter structure : Array(Int32)
        # property net_set : Array(MiniNet)
        # getter net_set_size : Int32
        # getter net_set_indexes_reversed : Array(Int32)
        # getter weight_height_mismatches : Array(Hash(Symbol, Int32))

        # def initialize(@net_set)
        #   @structure = calc_structure
        #   @net_set_size = @net_set.size
        #   @net_set_indexes_reversed = Array.new(@net_set_size) { |i| @net_set_size - i - 1}

        #   @weight_height_mismatches = Array(Hash(Symbol, Int32)).new
        # end

        # def validate
        #   index_max = @net_set_size - 1

        #   @weight_height_mismatches = Array(Hash(Symbol, Int32)).new
        #   @weight_height_mismatches = @net_set.map_with_index do |net_from, index|
        #     if index >= index_max
        #       nil # There is no 'next' net after the last net, so we don't need to compare any sizes
        #     else
        #       net_to = @net_set[index + 1]
        #       if net_from.width != net_to.height_considering_bias
        #         {
        #           :from_index                 => index,
        #           :to_index                   => index + 1,
        #           :from_width                 => net_from.width,
        #           :to_height_considering_bias => @net_set[index + 1].height_considering_bias,
        #           :from_disable_bias          => (net_from.disable_bias ? 1 : 0),
        #           :to_disable_bias            => (@net_set[index + 1].disable_bias ? 1 : 0),
        #         }
        #       end
        #     end
        #   end.compact

        #   @weight_height_mismatches.any? ? false : true
        # end

        # def errors
        #   @weight_height_mismatches
        # end

        # def validate!
        #   validate ? true : raise "Invalid net set (width vs height mismatch), errors: #{errors}"
        # end

        # def calc_structure
        #   @net_set.map do |net|
        #     net.height
        #   end << @net_set.last.width
        # end

        # def eval(inputs_given)
        #   @net_set.each_with_index do |net, index|
        #     if index == 0
        #       net.step_load_inputs(inputs_given)
        #     else
        #       net.step_load_inputs(@net_set[index - 1].outputs_guessed)
        #     end

        #     net.step_calc_forward
        #   end

        #   @net_set.last.outputs_guessed
        # end

        # # TODO: utilize until_min_avg_error
        # def train(inputs_given, outputs_expected, until_min_avg_error = 0.1)
        #   @net_set.each_with_index do |net, index|
        #     index == 0 ? net.step_load_inputs(inputs_given) : net.step_load_inputs(@net_set[index - 1].outputs_guessed)

        #     net.step_calc_forward
        #   end

        #   index_max = @net_set_size - 1
        #   @net_set_indexes_reversed.each do |index|
        #     net = @net_set[index]

        #     index == index_max ? net.step_load_outputs(outputs_expected) : net.step_load_outputs(@net_set[index + 1].input_deltas)

        #     net.step_calculate_error
        #     net.step_backpropagate
        #   end

        #   @net_set.last.error_total
        # end

        # def guesses_best
        #   @net_set.last.guesses_best
        # end

        # def step_calculate_error_distance_history
        #   @net_set.last.step_calculate_error_distance_history
        # end

        # def error_distance_history
        #   @net_set.last.error_distance_history
        # end
      end
    end
  end
end
