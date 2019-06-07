require "json"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Node
        struct Net
          include JSON::Serializable

          property config : Node::Config
          property bp_net : Backpropagation::Net

          def initialize(
            rnn_config : Rnn::Config,
            channel_set_index : Int32, channel_type : Int32, time_col_index : Int32,
            node_input_mappings : Array(NodeCoord)
          )
            @config = Node::Config.new(
              rnn_config,
              channel_set_index, channel_type, time_col_index,
              node_input_mappings
            )

            @bp_net = init_bp_net
          end

          
          def init_bp_net
            node_structure = [config.node_input_cache.flatten.size] + config.rnn_config.structure_hidden_laters + [config.rnn_config.qty_states_hidden_out]
            Backpropagation::Net.new(
              structure: node_structure,
              disable_bias: !(config.channel_set_index == 0),
              learning_rate: config.rnn_config.learning_rate,
              momentum: config.rnn_config.momentum
            )
          end
        end
      end
    end
  end
end
      