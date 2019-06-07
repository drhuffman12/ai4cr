require "json"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Node
        struct Net
          include JSON::Serializable

          property state : Node::State

          def initialize(
            rnn_config : Rnn::Config,
            channel_set_index : Int32, channel_type : Int32, time_col_index : Int32,
            node_input_mappings : Array(NodeCoord)
          )
            @state = Node::State.new(
              rnn_config,
              channel_set_index, channel_type, time_col_index,
              node_input_mappings
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
      