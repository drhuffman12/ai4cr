require "./math"

module Ai4cr
  module NeuralNetwork
    module Rnnbim # RNN, Bidirectional, Inversable Memory
      class Net # Ai4cr::NeuralNetwork::Rnnbim::Net

        NODE_VAL_MIN = -1.0
        NODE_VAL_MID = 0.0
        NODE_VAL_MAX = 1.0

        getter time_column_scale, time_column_qty : Int32, input_state_qty, output_state_qty
        getter hidden_layer_qty, hidden_layer_scale
        getter hidden_state_qty : Int32

        getter hidden_layer_range : Range(Int32, Int32)
        getter time_column_range : Range(Int32, Int32)
        getter input_state_range : Range(Int32, Int32)
        getter output_state_range : Range(Int32, Int32)
        getter hidden_layer_range : Range(Int32, Int32)
        getter hidden_state_range : Range(Int32, Int32)

        getter hidden_channel_keys : Array(Symbol) # TODO: change to Array(Enum)?
        getter hidden_offset_scales : Array(Int32)

        getter nodes_in : NodesChrono
        getter nodes_hidden : NodesHidden
        getter nodes_out : NodesChrono
        getter nodes_out_expected : NodesChrono # used in conjunction w/ nodes_out_required
        getter nodes_out_required_not : NodesChrono # Must NOT match: -1.0
        getter nodes_out_required_undecided : NodesChrono # Undecided/Don't Care: 0.0
        getter nodes_out_required_all : NodesChrono # Must Match: 1.0
        getter nodes_out_required : NodesChrono # values in -1.0 to 1.0

        # getter delta_in : NodesChrono, 
        getter delta_out : NodesChrono, delta_hidden : NodesHidden
        property network_weights : WeightsNetwork
        property network_weight_changes : WeightsNetwork

        # property current_inputs, current_outputs

        def initialize(
            @time_column_scale = 1, 
            @input_state_qty = 4, @output_state_qty = 2,
            @hidden_layer_qty = 2, @hidden_layer_scale = 1.0
            # ,
            # @local_offsets = [1,2,3,5,7,9,11] # TODO: e.g.: prime offsets
          )
          raise ArgumentError.new("The value of arg 'hidden_layer_qty' must be at least 1.") if hidden_layer_qty < 1

          @time_column_qty = 2 * (2 ** hidden_layer_qty) * time_column_scale
          @time_column_range = (0..time_column_qty-1)
          @input_state_range = (0..input_state_qty-1)
          @output_state_range = (0..output_state_qty-1)

          @hidden_layer_range = (0..hidden_layer_qty-1)
          @hidden_state_qty = ((input_state_qty + output_state_qty) * hidden_layer_scale / 2.0).ceil.to_i32
          @hidden_state_range = (0..hidden_state_qty-1)

          @hidden_channel_keys = [:past, :local, :future, :combo]

          @nodes_in = time_column_range.map { |t| input_state_range.map { |s| 0.0 } }
          @nodes_out = time_column_range.map { |t| output_state_range.map { |s| 0.0 } }
          
          @nodes_out_expected = time_column_range.map { |t| output_state_range.map { |s| 0.0 } }
          @nodes_out_required_not = time_column_range.map { |t| output_state_range.map { |s| -1.0 } }
          @nodes_out_required_undecided = time_column_range.map { |t| output_state_range.map { |s| 0.0 } }
          @nodes_out_required_all = time_column_range.map { |t| output_state_range.map { |s| 1.0 } }
          @nodes_out_required = nodes_out_required_all

          @nodes_hidden = init_hidden_nodes

          @hidden_offset_scales = hidden_layer_range.map { |l| 2 ** l }

          @network_weights = init_network_weights
          @network_weight_changes = init_network_weights
          
          @delta_out = time_column_range.map { |t| output_state_range.map { |s| 0.0 } }
          @delta_hidden = init_hidden_nodes
        end

        def init_hidden_nodes
          hn = NodesHidden.new
          hidden_layer_range.map do # |layer|
            node_sets = NodesChannel.new
            hidden_channel_keys.each do |channel_key|
              node_sets[channel_key] = time_column_range.map { |t| hidden_state_range.map { |s| 0.0 } }
            end
            hash = {
              :current => node_sets,
              :mem => node_sets.clone
            }
            hn << hash
          end
          hn
        end

        def init_network_weights          
          nw = WeightsNetwork.new

          # hidden layers
          hidden_layer_range.each do |hidden_layer_index|
            node_sets = WeightsToChannel.new # Hash(Symbol,Array(WeightsFromChannel)).new
            hidden_channel_keys.each do |channel_key|
              node_sets[:past] = time_column_range.map { |time_column_index| init_weights_to_current_past(time_column_index, hidden_layer_index) }
              node_sets[:local] = time_column_range.map { |time_column_index| init_weights_to_current_local(time_column_index, hidden_layer_index) }
              node_sets[:future] = time_column_range.map { |time_column_index| init_weights_to_current_future(time_column_index, hidden_layer_index) }
              node_sets[:combo] = time_column_range.map { |time_column_index| init_weights_to_current_combo(time_column_index, hidden_layer_index) }
            end
            nw["hidden_#{hidden_layer_index}"] = node_sets
          end

          # output layer
          channel_key = :output
          node_sets = WeightsToChannel.new
          node_sets[channel_key] = time_column_range.map do |time_column_index|
            # hidden_state_range.map do |s|
              # init_weights_for_output_current(time_column_index)
              # {
              #   :combo => [[0.0]],
              #   :bias => [[0.0]]
              # }
              init_weights_to_current_output(time_column_index)
            # end
          end
          nw["output"] = node_sets

          nw
        end

        def init_weights_to_current_past(time_column_index, hidden_layer_index)
          weights_from_channel = WeightsFromChannel.new

          if hidden_layer_index == 0
            weights_from_channel[:input] = init_weights_from_inputs_to_hidden
          else
            weights_from_channel[:combo] = init_weights_from_hidden_to_hidden
          end

          weights_from_channel[:past] = init_weights_from_hidden_to_hidden if time_column_index >= Math.node_scaled_border_past(hidden_layer_index)

          weights_from_channel[:mem] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          
          weights_from_channel
        end

        def init_weights_to_current_local(time_column_index, hidden_layer_index)
          weights_from_channel = WeightsFromChannel.new

          if hidden_layer_index == 0
            weights_from_channel[:input_past] = init_weights_from_inputs_to_hidden if time_column_index >= Math.node_scaled_border_past(hidden_layer_index)
            weights_from_channel[:input_current] = init_weights_from_inputs_to_hidden
            weights_from_channel[:input_future] = init_weights_from_inputs_to_hidden if time_column_index < Math.node_scaled_border_future(time_column_range, hidden_layer_index)
          else
            weights_from_channel[:combo_past] = init_weights_from_hidden_to_hidden if time_column_index >= Math.node_scaled_border_past(hidden_layer_index)
            weights_from_channel[:combo_current] = init_weights_from_hidden_to_hidden
            weights_from_channel[:combo_future] = init_weights_from_hidden_to_hidden if time_column_index < Math.node_scaled_border_future(time_column_range, hidden_layer_index)
          end

          weights_from_channel[:mem] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          
          weights_from_channel
        end

        def init_weights_to_current_combo(time_column_index, hidden_layer_index)
          weights_from_channel = WeightsFromChannel.new

          if hidden_layer_index == 0
            weights_from_channel[:input] = init_weights_from_inputs_to_hidden
          else
            weights_from_channel[:combo] = init_weights_from_hidden_to_hidden
          end

          weights_from_channel[:past] = init_weights_from_hidden_to_hidden
          weights_from_channel[:local] = init_weights_from_hidden_to_hidden
          weights_from_channel[:future] = init_weights_from_hidden_to_hidden
          
          weights_from_channel[:mem] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          
          weights_from_channel
        end

        def init_weights_to_current_future(time_column_index, hidden_layer_index)
          weights_from_channel = WeightsFromChannel.new

          if hidden_layer_index == 0
            weights_from_channel[:input] = init_weights_from_inputs_to_hidden
          else
            weights_from_channel[:combo] = init_weights_from_hidden_to_hidden
          end

          # weights_from_channel[:past] = init_weights_from_hidden_to_hidden if past_enabled && time_column_index > 0
          weights_from_channel[:future] = init_weights_from_hidden_to_hidden if time_column_index < Math.node_scaled_border_future(time_column_range, hidden_layer_index) # if time_column_index != time_column_range.max

          weights_from_channel[:mem] = init_weights_from_hidden_to_hidden
          weights_from_channel[:bias] = init_weights_from_bias_to_hidden
          weights_from_channel
        end

        def init_weights_to_current_output(time_column_index)
          weights_from_channel = WeightsFromChannel.new

          weights_from_channel[:combo] = init_weights_from_hidden_to_output
          weights_from_channel[:past] = init_weights_from_hidden_to_output
          weights_from_channel[:future] = init_weights_from_hidden_to_output
          weights_from_channel[:bias] = init_weights_from_bias_to_output

          weights_from_channel
        end

        def init_weights_from_inputs_to_hidden
          # time_column_range.map { |t| input_state_range.map { |i| hidden_state_range.map { |s| Math.rnd_pos_neg_one } } }
          input_state_range.map { |i| hidden_state_range.map { |s| Math.rnd_pos_neg_one } }
        end

        def init_weights_from_hidden_to_hidden
          hidden_state_range.map { |i| hidden_state_range.map { |s| Math.rnd_pos_neg_one } }
        end

        def init_weights_from_input_to_output
          input_state_range.map { |i| output_state_range.map { |s| Math.rnd_pos_neg_one } }
        end

        def init_weights_from_hidden_to_output
          hidden_state_range.map { |i| output_state_range.map { |s| Math.rnd_pos_neg_one } }
          # output_state_range.map { |i| hidden_state_range.map { |s| Math.rnd_pos_neg_one } }
        end

        def init_weights_from_bias_to_hidden
          [0].map { |i| hidden_state_range.map { |s| Math.rnd_pos_neg_one } }
        end

        def init_weights_from_bias_to_output
          [0].map { |i| output_state_range.map { |s| Math.rnd_pos_neg_one } }
        end

        def check_io_dimensions(inputs, outputs)
          @nodes_in = check_input_dimension(inputs)
          @nodes_out_expected = check_output_dimension(outputs)
        end

        def check_input_dimension(inputs)
          # current_inputs, current_outputs
          raise ArgumentError.new("Bad Inputs") if inputs.size != nodes_in.size || inputs.map{|i| i.size} != nodes_in.map{|i| i.size}
          inputs.map{|i| i.map{|j| j.to_f}}
        end

        def check_output_dimension(outputs)
          # current_inputs, current_outputs
          raise ArgumentError.new("Bad Outputs") if outputs.size != nodes_out.size || outputs.map{|i| i.size} != nodes_out.map{|i| i.size}
          outputs.map{|i| i.map{|j| j.to_f}}
        end

        def eval(inputs)
          @nodes_in = check_input_dimension(inputs)
          # eval_raw
          eval_sums
        end

        # def eval_raw
        #   @nodes_in = inputs.map{|i| i.map{|j| j.to_f}} # clone
        #   eval_sums
        # end

        def eval_sums
          # @network_weights

          # hidden layers
          hidden_layer_range.each do |hidden_layer_index|
            node_sets = WeightsToChannel.new # Hash(Symbol,Array(WeightsFromChannel)).new
            hidden_channel_keys.each do |channel_key|
              node_sets[:past] = time_column_range.map { |time_column_index| init_weights_to_current_past(time_column_index, hidden_layer_index) }
              node_sets[:local] = time_column_range.map { |time_column_index| init_weights_to_current_local(time_column_index, hidden_layer_index) }
              node_sets[:future] = time_column_range.map { |time_column_index| init_weights_to_current_future(time_column_index, hidden_layer_index) }
              node_sets[:combo] = time_column_range.map { |time_column_index| init_weights_to_current_combo(time_column_index, hidden_layer_index) }
            end
            nw["hidden_#{hidden_layer_index}"] = node_sets
          end

          # output layer
          channel_key = :output
          node_sets = WeightsToChannel.new
          node_sets[channel_key] = time_column_range.map do |time_column_index|
            # hidden_state_range.map do |s|
              # init_weights_for_output_current(time_column_index)
              # {
              #   :combo => [[0.0]],
              #   :bias => [[0.0]]
              # }
              init_weights_to_current_output(time_column_index)
            # end
          end
          nw["output"] = node_sets

        end

        def backpropagate(outputs)
        end

        def calculate_error(outputs)
        end

        def nodes_out_required(requirements)
          @nodes_out_required = requirements
        end

        def nodes_out_required_not!
          @nodes_out_required = @nodes_out_required_not
        end

        def nodes_out_required_undecided!
          @nodes_out_required = @nodes_out_required_undecided
        end

        def nodes_out_required_all!
          @nodes_out_required = @nodes_out_required_all
        end

        def train(inputs, outputs)
          @nodes_out_expected = check_output_dimension(outputs)
          eval(inputs)
          
          # inputs = inputs.map { |v| v.to_f }
          # outputs = outputs.map { |v| v.to_f }
          # check_io_dimensions(inputs, outputs)
          # eval_raw
          backpropagate(outputs)
          calculate_error(outputs)
        end
      end
    end
  end
end
