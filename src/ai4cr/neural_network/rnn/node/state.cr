require "json"
# require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Node
        struct State
          include JSON::Serializable

          property config : Node::Config
          property recent_memory : Array(Array(Float64))
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
            @recent_memory = init_recent_memory
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

          def init_recent_memory
            @config.rnn_config.qty_recent_memory.times.to_a.map do |recent_memory_index|
              @config.rnn_config.qty_states_hidden_out.times.to_a.map { |hidden_out_index| 0.0}
            end
          end

          def cycle_recent_memory
            @recent_memory.shift
            @recent_memory << @bp_net.state.activation_nodes.last.clone
          end
        end
      end
    end
  end
end
      
# # module Ai4cr
# #   module NeuralNetwork
# #     module Rnn
# #       struct Node
# #         include JSON::Serializable
        
# #         property cfg_output

# #         # include NeuralNetwork::Common::Initializers::StructureHiddenLaters
# #         # include NeuralNetwork::Common::Initializers::LearningRate
# #         # include NeuralNetwork::Common::Initializers::Momentum

# #         # property qty_states_in
# #         # property qty_states_out
# #         # property qty_time_cols
# #         # property qty_lpfc_layers
# #         # property qty_time_cols_neighbor_inputs

# #         # property structure_hidden_laters : Array(Int32)
# #         # property disable_bias : Bool

# #         property qty_inputs_from_memory : Int32
# #         property qty_inputs_from_other_nodes : Int32
# #         property qty_hidden_layers : Array(Int32)
# #         property qty_outputs : Int32
# #         property structure : Array(Int32)

# #         property net : NeuralNetwork::Backpropagation::Net
# #         property qty_recent_memory : Int32
# #         property recent_memory : Array(Array(Float64))

# #         def initialize(
# #           # # RNN Net:
# #           # @qty_states_in = 3,
# #           # @qty_states_out = 4,
# #           # @qty_time_cols = 5,
# #           # @qty_lpfc_layers = 3,
# #           # @qty_time_cols_neighbor_inputs = 2,
# #           # @qty_recent_memory = 2,

# #           # Embedded Backpropagation Nets:
# #           # @structure_hidden_laters = [] of Int32,
# #           # hidden_scale_factor = 2,

# #           structure_hidden_laters : Array(Int32)? = nil,
# #           disable_bias = true, learning_rate = nil, momentum = nil,

# #           # RNN Node:
# #           @qty_recent_memory = 3
# #         )
# #           # @structure_hidden_laters = init_structure_hidden_laters(structure_hidden_laters, qty_states_in, qty_states_out, hidden_scale_factor)
# #           # @disable_bias = !!disable_bias
# #           # @learning_rate = init_learning_rate(learning_rate)
# #           # @momentum = init_momentum(momentum)
# #           qty_inputs_from_memory = 
# #           _structure = 
# #           @net = init_net(structure_hidden_laters, disable_bias, learning_rate, momentum)
# #           @recent_memory = init_recent_memory
# #         end

# #         # def init_structure_hidden_laters(_structure_hidden_laters)
# #         #   # must be positive
# #         #   _structure_hidden_laters.nil? ? [qty_states_in + qty_states_out] : _structure_hidden_laters.as(Array(Int32))
# #         # end

# #         def init_net
# #           NeuralNetwork::Backpropagation::Net.new(
# #                   structure: _structure,
# #                   disable_bias: _disable_bias,
# #                   learning_rate: _learning_rate,
# #                   momentum: _momentum
# #                 )
# #         end

# #         def init_recent_memory
# #           qty_recent_memory.times.map{ @net.state.activation_nodes.last.clone }          
# #         end

# #         def cycle_recent_memory
# #           @recent_memory.shift
# #           @recent_memory << @net.state.activation_nodes.last.clone
# #         end
# #       end
# #     end
# #   end
# # end
