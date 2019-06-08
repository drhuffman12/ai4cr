require "json"
# require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module MemBkprop
        struct Config
          include JSON::Serializable

          property rnn_config : Rnn::Config
          property channel_set_index : Int32
          property channel_type : Int32
          property time_col_index : Int32
          property mem_bkprop_input_mappings : Array(MemBkpropCoord)
          property mem_bkprop_input_cache : Array(Array(Float64))

          def initialize(
            @rnn_config,
            @channel_set_index, @channel_type, @time_col_index,
            @mem_bkprop_input_mappings
          )
            @mem_bkprop_input_cache = init_mem_bkprop_input_cache
          end

          def init_mem_bkprop_input_cache
            mem_bkprop_input_mappings.map_with_index do |mem_bkprop_input_mapping, i|
              case mem_bkprop_input_mapping[:channel_type]
              when ChannelType::Local.value, ChannelType::Past.value, ChannelType::Future.value, ChannelType::Output.value, ChannelType::Combo.value, ChannelType::Memory.value
                rnn_config.qty_states_channel_out.times.to_a.map{ 0.0 }
              when ChannelType::Input.value
                rnn_config.qty_states_in.times.to_a.map{ 0.0 }

              # when ChannelType::Memory.value # TODO:
              #   # rnn_config.qty_recent_memory.times.to_a.map{ rnn_config.qty_states_channel_out.times.to_a.map{ 0.0 } }.flatten
              #   rnn_config.qty_states_channel_out.times.to_a.map{ 0.0 }

              else
                raise "Invalid input mapping type: #{mem_bkprop_input_mapping[:channel_type]}, i: #{i}, mem_bkprop_input_mapping: #{mem_bkprop_input_mapping}"
                # [] of Float64
              end
            end
          end   
        end
      end
    end
  end
end
