require "json"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module Node
        struct Net
          include JSON::Serializable

          property qty_states_in
          property qty_states_hidden_out
          property qty_states_out
          property qty_recent_memory

          property channel_set_index
          property channel_type
          property time_col_index

          # TODO: Move some of these to Node::Config
          property node_input_mappings
          property node_input_cache : Array(Array(Float64))

          def initialize(
            @qty_states_in : Int32, @qty_states_hidden_out : Int32, @qty_states_out : Int32, @qty_recent_memory : Int32,
            @channel_set_index : Int32, @channel_type : Int32, @time_col_index : Int32,
            @node_input_mappings : Array(NodeCoord),
          )
            @node_input_cache = init_node_input_cache
          end

          def init_node_input_cache
            node_input_mappings.map_with_index do |node_input_mapping, i|
              case node_input_mapping[:channel_type]
              when ChannelType::Local.value, ChannelType::Past.value, ChannelType::Future.value, ChannelType::Output.value, ChannelType::Combo.value, ChannelType::Memory.value
                qty_states_hidden_out.times.to_a.map{ 0.0 }
              when ChannelType::Input.value
                qty_states_in.times.to_a.map{ 0.0 }

              # when ChannelType::Output.value # TODO:
              #   qty_states_out.times.to_a.map{ 0.0 }

              # when ChannelType::Memory.value # TODO:
              #   # qty_recent_memory.times.to_a.map{ qty_states_hidden_out.times.to_a.map{ 0.0 } }.flatten
              #   qty_states_hidden_out.times.to_a.map{ 0.0 }

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
      