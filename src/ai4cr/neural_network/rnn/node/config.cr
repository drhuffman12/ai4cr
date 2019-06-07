require "json"
# require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Node
        struct Config
          include JSON::Serializable

          property rnn_config : Rnn::Config
          property channel_set_index : Int32
          property channel_type : Int32
          property time_col_index : Int32
          property node_input_mappings : Array(NodeCoord)
          property node_input_cache : Array(Array(Float64))

          def initialize(
            @rnn_config,
            @channel_set_index, @channel_type, @time_col_index,
            @node_input_mappings
          )
            @node_input_cache = init_node_input_cache
          end

          def init_node_input_cache
            node_input_mappings.map_with_index do |node_input_mapping, i|
              case node_input_mapping[:channel_type]
              when ChannelType::Local.value, ChannelType::Past.value, ChannelType::Future.value, ChannelType::Output.value, ChannelType::Combo.value, ChannelType::Memory.value
                rnn_config.qty_states_hidden_out.times.to_a.map{ 0.0 }
              when ChannelType::Input.value
                rnn_config.qty_states_in.times.to_a.map{ 0.0 }

              # when ChannelType::Memory.value # TODO:
              #   # rnn_config.qty_recent_memory.times.to_a.map{ rnn_config.qty_states_hidden_out.times.to_a.map{ 0.0 } }.flatten
              #   rnn_config.qty_states_hidden_out.times.to_a.map{ 0.0 }

              else
                raise "Invalid input mapping type: #{node_input_mapping[:channel_type]}, i: #{i}, node_input_mapping: #{node_input_mapping}"
                # [] of Float64
              end
            end
          end   
        end
      end
    end
  end
end
