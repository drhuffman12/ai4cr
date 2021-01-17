require "json"
require "./rnn_simple.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      class RnnSimpleTeam
        # Team of Simple RNN's w/ inputs, hidden forward-feeding recurrent layer(s), outputs, and some other params
        include JSON::Serializable

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

        def initialize(
          @team_size = 10,

          @io_offset = RnnSimple::IO_OFFSET_DEFAULT,
          @time_col_qty = RnnSimple::TIME_COL_QTY_MIN,
          @input_size = RnnSimple::INPUT_SIZE_MIN,
          @output_size = RnnSimple::OUTPUT_SIZE_MIN,
          @hidden_size_given = nil,
          @hidden_layer_qty = RnnSimple::HIDDEN_LAYER_QTY_MIN,

          @learning_style : LearningStyle = LS_RELU
        )
          raise "Size error; team_size must be > 0." if team_size <= 0

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
            )
          end
        end
      end
    end
  end
end
