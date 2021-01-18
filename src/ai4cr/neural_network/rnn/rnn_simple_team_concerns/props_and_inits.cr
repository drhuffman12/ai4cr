module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnSimpleTeamConcerns
        module PropsAndInits
          property team_size : Int32
          property team_members : Array(RnnSimple)

          getter io_offset : Int32
          getter time_col_qty : Int32

          getter input_size : Int32
          getter output_size : Int32
          getter hidden_size_given : Int32?
          getter hidden_layer_qty : Int32
          # getter hidden_size : Int32

          property disable_bias : Bool
          property bias_default : Float64

          property learning_style : LearningStyle

          def initialize(
            @team_size = 10,

            @io_offset = RnnSimple::IO_OFFSET_DEFAULT,
            @time_col_qty = RnnSimple::TIME_COL_QTY_MIN,
            @input_size = RnnSimple::INPUT_SIZE_MIN,
            @output_size = RnnSimple::OUTPUT_SIZE_MIN,
            @hidden_size_given = nil,
            @hidden_layer_qty = RnnSimple::HIDDEN_LAYER_QTY_MIN,

            disable_bias : Bool? = nil,
            @bias_default = 1.0,

            @learning_style : LearningStyle = LS_RELU
          )
            raise "Size error; team_size must be > 0." if team_size <= 0

            # TODO: Handle differing hidden layer output sizes
            if hidden_size_given.is_a?(Int32)
              @hidden_size = @hidden_size_given.as(Int32)
            else
              @hidden_size = @input_size + @output_size
            end

            # TODO: switch 'disabled_bias' to 'enabled_bias' and adjust defaulting accordingly
            @disable_bias = disable_bias.nil? ? false : disable_bias

            @team__indexes = Array(Int32).new(team_size) { |i| i }
            @team_members = @team__indexes.map do #  |i|
              RnnSimple.new(
                io_offset: @io_offset,
                time_col_qty: @time_col_qty,
                input_size: @input_size,
                output_size: @output_size,
                hidden_size_given: @hidden_size_given,
                hidden_layer_qty: @hidden_layer_qty,

                disable_bias: @disable_bias,
                bias_default: @bias_default,

                learning_style: @learning_style,
              )
            end
          end

          def hidden_size
            team_members.first.hidden_size
          end
        end
      end
    end
  end
end