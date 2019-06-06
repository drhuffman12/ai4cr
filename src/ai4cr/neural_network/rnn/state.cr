require "json"
require "../common"

module Ai4cr
  module NeuralNetwork
    module Rnn
      struct State
        include JSON::Serializable
        # include Ai4cr::NeuralNetwork::Backpropagation::Math

        # enum ChannelType
        #   Local
        #   Past
        #   Future
        #   Combo
        #   Input
        #   Output
        # end

        # # alias NodeCoord = NamedTuple(channel_set_index: Int32, channel_type: ChannelType, time_col_index: Int32)
        # alias NodeCoord = NamedTuple(channel_set_index: Int32, channel_type: Int32, time_col_index: Int32)

        property config : Config
        # property channel_set : Array(ChannelSet)

        # property node_input_mappings : Array(Hash(Symbol, Array(NodeCoord)))
        # property node_input_mappings : Array(Array(Array(Array(Rnn::NodeCoord))))
        # property node_input_mappings : NamedTuple(
        #   channel_sets: Array(
        #     NamedTuple(
        #       channel_types: Array(
        #         NamedTuple(
        #           time_col_indexes: Array(
        #             # Array(NamedTuple(channel_set_index: Int32, channel_type: Int32, time_col_index: Int32))
        #             Array(Rnn::NodeCoord)
        #           )
        #         )
        #       )
        #     )
        #   )
        # )


        # property nodes : Array(Array(Hash(Symbol, Array(NeuralNetwork::Backpropagation::Net))))
        # property nodes : Hash(NodeCoord, NeuralNetwork::Backpropagation::Net)
        property nodes : Array(Array(Array(Ai4cr::NeuralNetwork::Rnn::Node::Net)))
        # property nodes : Array(Hash(Symbol, Array(NeuralNetwork::Backpropagation::Net)))

        # property node_input_mappings 
        # property node_input_mappings : # time_col_index => channel_set_index => channel_symbol => 

        # include NeuralNetwork::Common::Initializers::LearningRate
        # include NeuralNetwork::Common::Initializers::Momentum

        # property qty_states_in
        # property qty_states_out
        # property qty_time_cols
        # property qty_lpfc_layers
        # property qty_recent_memory

        # property structure_hidden_laters : Array(Int32)
        # property disable_bias : Bool

        # property inputs  : Array(Array(Float64))
        # property outputs : Array(Array(Float64))

        # property local_channels  : Array(Array(NeuralNetwork::Backpropagation::Net))
        # property past_channels   : Array(Array(NeuralNetwork::Backpropagation::Net))
        # property future_channels : Array(Array(NeuralNetwork::Backpropagation::Net))
        # property combo_channels  : Array(Array(NeuralNetwork::Backpropagation::Net))

        def initialize(
          # RNN Net:
          qty_states_in = 3, qty_states_hidden_out = 5, qty_states_out = 4,
          qty_time_cols = 5,
          qty_lpfc_layers = 3,
          qty_hidden_laters = 2,
          qty_time_cols_neighbor_inputs = 2,
          qty_recent_memory = 2,

          # Embedded Backpropagation Nets:
          # @hidden_scale_factor = 2,
          # structure_hidden_laters : Array(Int32)? = nil, disable_bias = true, 
          learning_rate = nil, momentum = nil
        )
          @config = Config.new(
            qty_states_in, qty_states_hidden_out, qty_states_out,
            qty_time_cols,
            qty_lpfc_layers, qty_hidden_laters,
            qty_time_cols_neighbor_inputs, qty_recent_memory,
            # hidden_scale_factor, 
            # structure_hidden_laters, 
            # disable_bias, 
            learning_rate, momentum
          )

          # @inputs = init_inputs
          # @outputs = init_outputs
          # @channel_set = init_nodes # (config) # (disable_bias)

          # @node_input_mappings = init_node_input_mappings
          @nodes = init_nodes # (config) # (disable_bias)
          # @io_connections = init_io_connections
        end

        # def init_node_input_mappings
        #   # _node_input_mappings = Array(Hash(Symbol, Array(NodeCoord))).new

        #   # qty_states_in_and_out = config.qty_states_out + config.qty_states_out

        #   # _learning_rate = config.learning_rate
        #   # _momentum = config.momentum
          
        #   # channel_set_index_max = config.qty_lpfc_layers - 1

        #   # (0..(config.qty_lpfc_layers - 1)).to_a.map do |channel_set_index|
        #   #   [ChannelType::Local.value,ChannelType::Past.value,ChannelType::Future.value,ChannelType::Combo.value].map do |channel_type|
        #   #     (0..(config.qty_time_cols - 1)).to_a.map do |time_col_index|                
        #   #       # _disable_bias = channel_set_index > 0 # TODO: utilize disable_bias .. but only if ????
                
        #   #       init_connections_to_node_at(channel_set_index, channel_type, time_col_index)
                
        #   #     end
        #   #   end
        #   # end

        #   {
        #     channel_sets: (0..(config.qty_lpfc_layers - 1)).to_a.map do |channel_set_index|
        #       {
        #         channel_types: [ChannelType::Local.value,ChannelType::Past.value,ChannelType::Future.value,ChannelType::Combo.value].map do |channel_type|
        #           {
        #             time_col_indexes: (0..(config.qty_time_cols - 1)).to_a.map do |time_col_index|                
        #               # _disable_bias = channel_set_index > 0 # TODO: utilize disable_bias .. but only if ????
                      
        #               init_connections_to_node_at(channel_set_index, channel_type, time_col_index)
                      
        #             end
        #           }
        #         end
        #       }
        #     end
        #   }
        # end

        def init_connections_to_node_at(channel_set_index, channel_type, time_col_index) : Array(NodeCoord)
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
          connections : Array(NodeCoord)
          connections = case channel_type
          when ChannelType::Local.value
            init_connections_to_node_at_channel_set_local(channel_set_index, prev_channel, time_col_index, time_col_indexes_before, time_col_indexes_after)
          when ChannelType::Past.value
            init_connections_to_node_at_channel_set_past(channel_set_index, prev_channel, time_col_index, time_col_indexes_before)
          when ChannelType::Future.value
            init_connections_to_node_at_channel_set_future(channel_set_index, prev_channel, time_col_index, time_col_indexes_after)
          else
            # Array(NodeCoord).new
            init_connections_to_node_at_channel_set_combo(channel_set_index, prev_channel, time_col_index)
          end

          if channel_set_index == 0 && channel_type == 0 && time_col_index == 0
            drh_debug_data = {
              channel_set_index: channel_set_index,
              channel_type: channel_type,
              time_col_index: time_col_index,
              qty_time_cols_neighbor_inputs: config.qty_time_cols_neighbor_inputs,
              qty_time_cols: config.qty_time_cols,
              time_col_index_left: time_col_index_left,
              time_col_index_right: time_col_index_right,
              time_col_indexes_before: time_col_indexes_before,
              time_col_indexes_after: time_col_indexes_after,
            }
            puts
            puts "drh_debug_data: #{drh_debug_data}"
            puts
          end

          connections
        end

        def init_connections_to_node_at_channel_set_local(channel_set_index, prev_channel, time_col_index, time_col_indexes_before, time_col_indexes_after) : Array(NodeCoord)
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
          end
        end

        def init_connections_to_node_at_channel_set_past(channel_set_index, prev_channel, time_col_index, time_col_indexes_before) : Array(NodeCoord)
          (
            time_col_indexes_before.map do |tci|
              {
                channel_set_index: time_col_index,
                channel_type: ChannelType::Past.value,
                time_col_index: tci
              }
            end +

            [{
              channel_set_index: time_col_index - 1,
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
          end
        end

        def init_connections_to_node_at_channel_set_future(channel_set_index, prev_channel, time_col_index, time_col_indexes_after) : Array(NodeCoord)
          (
            [{
              channel_set_index: time_col_index - 1,
              channel_type: prev_channel,
              time_col_index: time_col_index
            }] +

            time_col_indexes_after.map do |tci|
              {
                channel_set_index: time_col_index,
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
          end
        end

        def init_connections_to_node_at_channel_set_combo(channel_set_index, prev_channel, time_col_index) : Array(NodeCoord)
          [{
            channel_set_index: time_col_index - 1,
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
          }]
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

        # def init_nodes_OLD # (disable_bias)
        #   # alias NodeCoord = NamedTuple(time_col_index: Int32, channel_set_index: Int32, channel_type: Symbol)
        #   # property nodes : Hash(NodeCoord, NeuralNetwork::Backpropagation::Net)

        #   qty_states_in_and_out = config.qty_states_out + config.qty_states_out

        #   _learning_rate = config.learning_rate
        #   _momentum = config.momentum
          
        #   channel_set_index_max = config.qty_lpfc_layers - 1

        #   _channel_sets = Hash(NodeCoord, NeuralNetwork::Backpropagation::Net).new
        #   (0..channel_set_index_max).each do |channel_set_index|
        #     [ChannelType::Local.value, ChannelType::Past.value, ChannelType::Future.value, ChannelType::Combo.value].each do |channel_type|
        #       (0..(config.qty_time_cols - 1)).each do |time_col_index|
        #         if channel_set_index == 0
        #           _structure = [config.qty_states_in] + config.structure_hidden_laters + [qty_states_in_and_out]
        #           _disable_bias = false # TODO: utilize disable_bias .. but only if ????
        #         elsif channel_set_index == channel_set_index_max
        #           _structure = [qty_states_in_and_out] + config.structure_hidden_laters + [config.qty_states_out]
        #           _disable_bias = true # TODO: utilize disable_bias .. but only if ????
        #         else
        #           _structure = [qty_states_in_and_out] + config.structure_hidden_laters + [qty_states_in_and_out]
        #           _disable_bias = true # TODO: utilize disable_bias .. but only if ????
        #         end
                
        #         _disable_bias = channel_set_index > 0 # TODO: utilize disable_bias .. but only if ????

        #         node_coord = {
        #           channel_set_index: channel_set_index,
        #           channel_type: channel_type,
        #           time_col_index: time_col_index
        #         }
        #         _channel_sets[node_coord] = NeuralNetwork::Backpropagation::Net.new(
        #           structure: _structure,
        #           disable_bias: _disable_bias,
        #           learning_rate: _learning_rate,
        #           momentum: _momentum
        #         )

                
        #       end
        #     end
        #   end
        #   _channel_sets
        # end

        def init_nodes
          # _node_input_mappings = Array(Hash(Symbol, Array(NodeCoord))).new
          # _nodes = Hash(NodeCoord, NeuralNetwork::Backpropagation::Net).new
          # _nodes = Hash(NodeCoord, NeuralNetwork::Rnn::Node::Net).new

          channel_set_index_max = config.qty_lpfc_layers - 1

          # {
          #   channel_sets: (0..(config.qty_lpfc_layers - 1)).to_a.map do |channel_set_index|
          #     {
          #       channel_types: [ChannelType::Local.value,ChannelType::Past.value,ChannelType::Future.value,ChannelType::Combo.value].map do |channel_type|
          #         {
          #           time_col_indexes: (0..(config.qty_time_cols - 1)).to_a.map do |time_col_index|                
          #             # _disable_bias = channel_set_index > 0 # TODO: utilize disable_bias .. but only if ????
                      
          #             init_connections_to_node_at(channel_set_index, channel_type, time_col_index)
                      
          #           end
          #         }
          #       end
          #     }
          #   end
          # }

          (0..channel_set_index_max).map do |channel_set_index|
            [ChannelType::Local.value, ChannelType::Past.value, ChannelType::Future.value, ChannelType::Combo.value].map do |channel_type|
              (0..(config.qty_time_cols - 1)).map do |time_col_index|                
                # _disable_bias = channel_set_index > 0 # TODO: utilize disable_bias .. but only if ????
                
                init_node_at(channel_set_index, channel_type, time_col_index)
                
              end
            end
          end
        end
                
        def init_node_at(channel_set_index, channel_type, time_col_index) # coord : NodeCoord, config : NodeConfig) : Node
          # qty_states_in_and_out = config.qty_states_out + config.qty_states_out

          # _learning_rate = config.learning_rate
          # _momentum = config.momentum
          # node_input_mappings = node_input_mappings[channel_set_index][channel_type][time_col_index]
          # node_input_mappings = node_input_mappings[:channel_sets][channel_set_index][:channel_types][channel_type][:time_col_indexes][time_col_index]


          node_input_mappings = init_connections_to_node_at(channel_set_index, channel_type, time_col_index)
          
          NeuralNetwork::Rnn::Node::Net.new(
            @config.qty_states_in, @config.qty_states_hidden_out, @config.qty_states_out,
            channel_set_index, channel_type, time_col_index,
            node_input_mappings
          ) # (channel_set_index, channel_type, time_col_index, node_input_mappings)
        end

      end
    end
  end
end

# touch spec/ai4cr/neural_network/rnn/config_spec.cr