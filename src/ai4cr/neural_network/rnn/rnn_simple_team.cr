require "json"
# require "./../learning_style.cr"
require "./rnn_simple.cr"

# require "./rnn_concerns/calc_guess.cr"
# require "./rnn_concerns/props_and_inits.cr"
# require "./rnn_concerns/train_and_adjust.cr"
# require "./rnn_concerns/roll_ups.cr"
# require "./rnn_concerns/data_utils.cr"
# require "./rnn_concerns/training_utils.cr" # TODO!

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimpleTeam
        # Simple RNN w/ inputs, hidden forward-feeding recurrent layer(s), outputs, and some other params
        include JSON::Serializable

        # include RnnConcerns::PropsAndInits
        # include RnnConcerns::CalcGuess
        # include RnnConcerns::TrainAndAdjust
        # include RnnConcerns::RollUps
        # include RnnConcerns::DataUtils
        # include RnnConcerns::TrainingUtils # TODO!

        property team_size : Int32
        property team_members : Array(RnnSimple)

        getter io_offset : Int32
        getter time_col_qty : Int32
        getter input_size : Int32
        getter output_size : Int32
        # getter hidden_size : Int32
        getter hidden_size_given : Int32?
        getter hidden_layer_qty : Int32

        property learning_style : LearningStyle
        property deriv_scale : Float64
        property disable_bias : Bool
        property bias_default : Float64
        property learning_rate : Float64
        property momentum : Float64

        def initialize(
          @team_size = 10,

          @io_offset = RnnSimple::IO_OFFSET_DEFAULT,
          @time_col_qty = RnnSimple::TIME_COL_QTY_MIN,
          @input_size = RnnSimple::INPUT_SIZE_MIN,
          @output_size = RnnSimple::OUTPUT_SIZE_MIN,
          @hidden_size_given = nil,
          @hidden_layer_qty = RnnSimple::HIDDEN_LAYER_QTY_MIN,

          @learning_style : LearningStyle = LS_RELU,

          # for Prelu
          # TODO: set deriv_scale based on ?
          # @deriv_scale = 0.1,
          # @deriv_scale = 0.01,
          # @deriv_scale = 0.001,
          @deriv_scale = rand / 2.0,

          disable_bias : Bool? = nil, @bias_default = 1.0,

          learning_rate : Float64? = nil, momentum : Float64? = nil
          # _error_distance_history_max_ : Int32 = 10
        )
          raise "Size error; team_size must be > 0." if team_size <= 0

          # TODO: switch 'disabled_bias' to 'enabled_bias' and adjust defaulting accordingly
          @disable_bias = disable_bias.nil? ? false : disable_bias

          @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? rand : learning_rate.as(Float64)
          @momentum = momentum && momentum.as(Float64) > 0.0 ? momentum.as(Float64) : rand

          # @synaptic_layer_qty = hidden_layer_qty + 1

          # TODO: Handle differing hidden layer output sizes
          if hidden_size_given.is_a?(Int32)
            @hidden_size = @hidden_size_given.as(Int32)
          else
            @hidden_size = @input_size + @output_size
          end

          @team__indexes = Array(Int32).new(team_size) { |i| i }
          @team_members = @team__indexes.map do #  |i|
            RnnSimple.new(
              io_offset: @io_offset,
              time_col_qty: @time_col_qty,
              input_size: @input_size,
              output_size: @output_size,
              hidden_size_given: @hidden_size_given,
              hidden_layer_qty: @hidden_layer_qty,

              learning_style: @learning_style,

              # for Prelu
              # TODO: set deriv_scale based on ?
              # @deriv_scale = 0.1,
              # @deriv_scale = 0.01,
              # @deriv_scale = 0.001,
              deriv_scale: @deriv_scale,

              disable_bias: @disable_bias, bias_default: @bias_default,

              learning_rate: @learning_rate, momentum: @momentum
            )
          end
        end
      end
    end
  end
end
