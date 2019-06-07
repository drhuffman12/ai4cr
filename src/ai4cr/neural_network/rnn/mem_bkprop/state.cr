require "json"
# require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module MemBkprop
        struct State
          include JSON::Serializable

          property config : MemBkprop::Config
          property recent_memory : Array(Array(Float64))
          property bp_net : Backpropagation::Net

          def initialize(
            rnn_config,
            channel_set_index, channel_type, time_col_index,
            node_input_mappings
          )
            @config = MemBkprop::Config.new(
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

          def input=()
          end

          def init_recent_memory
            @config.rnn_config.qty_recent_memory.times.to_a.map do |recent_memory_index|
              @config.rnn_config.qty_states_hidden_out.times.to_a.map { |hidden_out_index| 0.0}
            end
          end

          def cycle_recent_memory
            @recent_memory.shift
            @recent_memory << output
          end

          def output
            @bp_net.state.activation_nodes.last.clone
          end
        end
      end
    end
  end
end
