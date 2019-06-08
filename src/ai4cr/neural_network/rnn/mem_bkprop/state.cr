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
            mem_bkprop_input_mappings
          )
            @config = MemBkprop::Config.new(
              rnn_config,
              channel_set_index, channel_type, time_col_index,
              mem_bkprop_input_mappings
            )
            @recent_memory = init_recent_memory
            @bp_net = init_bp_net
          end
          
          def init_bp_net
            mem_bkprop_structure = [config.mem_bkprop_input_cache.flatten.size] + config.rnn_config.structure_hidden_laters + [config.rnn_config.qty_states_channel_out]
            Backpropagation::Net.new(
              structure: mem_bkprop_structure,
              disable_bias: !(config.channel_set_index == 0),
              learning_rate: config.rnn_config.learning_rate,
              momentum: config.rnn_config.momentum
            )
          end

          def input=()
          end

          def init_recent_memory
            @config.rnn_config.qty_recent_memory.times.to_a.map do |recent_memory_index|
              @config.rnn_config.qty_states_channel_out.times.to_a.map { |hidden_out_index| 0.0}
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
