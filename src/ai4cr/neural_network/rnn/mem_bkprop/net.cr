require "json"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module MemBkprop
        struct Net
          include JSON::Serializable

          property state : MemBkprop::State

          def initialize(
            rnn_config : Rnn::Config,
            channel_set_index : Int32, channel_type : Int32, time_col_index : Int32,
            mem_bkprop_input_mappings : Array(MemBkpropCoord)
          )
            @state = MemBkprop::State.new(
              rnn_config,
              channel_set_index, channel_type, time_col_index,
              mem_bkprop_input_mappings
            )
          end

          # def collect_inputs
          #   # @inputs = ...
          # end

          # # def inputs
          # #   @inputs = ...
          # # end

          # def outputs
          # end

          # def guess_forward # (inputs)
          # end

          # def train_backwards # (outputs)
          # end
        end
      end
    end
  end
end
      