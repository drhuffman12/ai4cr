require "json"
require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      struct State
        include JSON::Serializable
        # include Ai4cr::NeuralNetwork::Backpropagation::Math

        property config : Config
        
        property mem_bkprops : Array(Array(Array(Ai4cr::NeuralNetwork::Rnn::MemBkprop::Net)))
        # property last_channel_to_outputs : Array(NeuralNetwork::Backpropagation::Net)
        property last_combos_to_outputs : Array(NeuralNetwork::Backpropagation::Net)

        def initialize(
          # RNN Net:
          qty_states_in = 3, qty_states_channel_out = 5, qty_states_out = 4,
          qty_time_cols = 5,
          qty_lpfc_layers = 3,
          qty_hidden_laters = 2,
          qty_time_cols_neighbor_inputs = 2,
          qty_recent_memory = 2,
          
          # Embedded Backpropagation Nets:
          learning_rate = nil, momentum = nil
        )
          @config = Config.new(
            # RNN Net:
            qty_states_in, qty_states_channel_out, qty_states_out,
            qty_time_cols,
            qty_lpfc_layers, qty_hidden_laters,
            qty_time_cols_neighbor_inputs, qty_recent_memory,

            # Embedded Backpropagation Nets:
            learning_rate, momentum
          )

          @mem_bkprops = init_mem_bkprops
          @last_combos_to_outputs = init_last_combos_to_outputs
        end

        def init_mem_bkprops
          channel_set_index_max = config.qty_lpfc_layers - 1
          (0..channel_set_index_max).map do |channel_set_index|
            [ChannelType::Local.value, ChannelType::Past.value, ChannelType::Future.value, ChannelType::Combo.value].map do |channel_type|
              (0..(config.qty_time_cols - 1)).map do |time_col_index|                
                init_mem_bkprop_at(channel_set_index, channel_type, time_col_index)
              end
            end
          end
        end
                
        def init_mem_bkprop_at(channel_set_index, channel_type, time_col_index)
          mem_bkprop_input_mappings = init_connections_to_mem_bkprop_at(channel_set_index, channel_type, time_col_index)
          NeuralNetwork::Rnn::MemBkprop::Net.new(
            @config,
            channel_set_index, channel_type, time_col_index,
            mem_bkprop_input_mappings
          )
        end

        def init_connections_to_mem_bkprop_at(channel_set_index, channel_type, time_col_index) : Array(MemBkpropCoord)
          time_col_index_left = (time_col_index - @config.qty_time_cols_neighbor_inputs)
          time_col_index_left = time_col_index_left < 0 ? 0 : time_col_index_left

          time_col_index_right = (time_col_index + @config.qty_time_cols_neighbor_inputs)
          time_col_index_right = time_col_index_right >= @config.qty_time_cols ? @config.qty_time_cols - 1 : time_col_index_right

          time_col_indexes_before = (time_col_index_left..(time_col_index - 1)).to_a
          time_col_indexes_after = ((time_col_index + 1)..time_col_index_right).to_a

          prev_channel = if channel_set_index == 0
            ChannelType::Input.value
          else
            ChannelType::Combo.value
          end
          connections : Array(MemBkpropCoord)
          connections = case channel_type
          when ChannelType::Local.value
            init_connections_to_mem_bkprop_at_channel_set_local(channel_set_index, prev_channel, time_col_index, time_col_indexes_before, time_col_indexes_after)
          when ChannelType::Past.value
            init_connections_to_mem_bkprop_at_channel_set_past(channel_set_index, prev_channel, time_col_index, time_col_indexes_before)
          when ChannelType::Future.value
            init_connections_to_mem_bkprop_at_channel_set_future(channel_set_index, prev_channel, time_col_index, time_col_indexes_after)
          else
            init_connections_to_mem_bkprop_at_channel_set_combo(channel_set_index, prev_channel, time_col_index)
          end

          connections
        end

        def init_connections_to_mem_bkprop_at_channel_set_local(channel_set_index, prev_channel, time_col_index, time_col_indexes_before, time_col_indexes_after) : Array(MemBkpropCoord)
          (
            time_col_indexes_before.map do |tci|
              {
                channel_set_index: channel_set_index - 1,
                channel_type: prev_channel,
                time_col_index: tci
              }
            end +

            [{
              channel_set_index: channel_set_index - 1,
              channel_type: prev_channel,
              time_col_index: time_col_index
            }] +

            time_col_indexes_after.map do |tci|
              {
                channel_set_index: channel_set_index - 1,
                channel_type: prev_channel,
                time_col_index: tci
              }
            end
          ).tap do |arr|
            if channel_set_index > 0
              arr << {
                channel_set_index: channel_set_index - 1,
                channel_type: ChannelType::Local.value,
                time_col_index: time_col_index
              }
            end
          end + 
          config.qty_recent_memory.times.to_a.map {
            {
              channel_set_index: channel_set_index,
              channel_type: ChannelType::Memory.value,
              time_col_index: time_col_index
            } 
          }
        end

        def init_connections_to_mem_bkprop_at_channel_set_past(channel_set_index, prev_channel, time_col_index, time_col_indexes_before) : Array(MemBkpropCoord)
          (
            time_col_indexes_before.map do |tci|
              {
                channel_set_index: channel_set_index,
                channel_type: ChannelType::Past.value,
                time_col_index: tci
              }
            end +

            [{
              channel_set_index: channel_set_index - 1,
              channel_type: prev_channel,
              time_col_index: time_col_index
            }]
          ).tap do |arr|
            if channel_set_index > 0
              arr << {
                channel_set_index: channel_set_index - 1,
                channel_type: ChannelType::Past.value,
                time_col_index: time_col_index
              }
            end
          end + 
          config.qty_recent_memory.times.to_a.map {
            {
              channel_set_index: channel_set_index,
              channel_type: ChannelType::Memory.value,
              time_col_index: time_col_index
            } 
          }
        end

        def init_connections_to_mem_bkprop_at_channel_set_future(channel_set_index, prev_channel, time_col_index, time_col_indexes_after) : Array(MemBkpropCoord)
          (
            [{
              channel_set_index: channel_set_index - 1,
              channel_type: prev_channel,
              time_col_index: time_col_index
            }] +

            time_col_indexes_after.map do |tci|
              {
                channel_set_index: channel_set_index,
                channel_type: ChannelType::Future.value,
                time_col_index: tci
              }
            end
          ).tap do |arr|
            if channel_set_index > 0
              arr << {
                channel_set_index: channel_set_index - 1,
                channel_type: ChannelType::Future.value,
                time_col_index: time_col_index
              }
            end
          end + 
          config.qty_recent_memory.times.to_a.map {
            {
              channel_set_index: channel_set_index,
              channel_type: ChannelType::Memory.value,
              time_col_index: time_col_index
            } 
          }
        end

        def init_connections_to_mem_bkprop_at_channel_set_combo(channel_set_index, prev_channel, time_col_index) : Array(MemBkpropCoord)
          [{
            channel_set_index: channel_set_index - 1,
            channel_type: prev_channel,
            time_col_index: time_col_index
          }] +
          [{
            channel_set_index: channel_set_index,
            channel_type: ChannelType::Local.value,
            time_col_index: time_col_index
          }] +
          [{
            channel_set_index: channel_set_index,
            channel_type: ChannelType::Past.value,
            time_col_index: time_col_index
          }] +
          [{
            channel_set_index: channel_set_index,
            channel_type: ChannelType::Future.value,
            time_col_index: time_col_index
          }] + 
          config.qty_recent_memory.times.to_a.map {
            {
              channel_set_index: channel_set_index,
              channel_type: ChannelType::Memory.value,
              time_col_index: time_col_index
            } 
          }
        end
          # .tap do |arr|
          #   if channel_set_index > 0
          #     arr << {
          #       channel_set_index: channel_set_index - 1,
          #       channel_type: ChannelType::Combo.value,
          #       time_col_index: time_col_index
          #     }
          #   end
          # end


        def init_last_combos_to_outputs
          channel_set_index_max = config.qty_lpfc_layers - 1
          channel_type = ChannelType::Combo.value
          (0..(config.qty_time_cols - 1)).map do |time_col_index|
            NeuralNetwork::Backpropagation::Net.new(
              structure: [config.qty_states_channel_out, config.qty_states_out],
              disable_bias: true,
              learning_rate: config.learning_rate,
              momentum: config.momentum
            )
          end
        end
      end
    end
  end
end
