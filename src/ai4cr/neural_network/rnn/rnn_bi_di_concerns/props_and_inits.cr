# require "../rnn_simple_concerns/props_and_inits.cr"

module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnBiDiConcerns
        module PropsAndInits
          # include RnnSimpleConcerns::PropsAndInits
          # # add/update/implement reverse time-col direction
          # def init_network_mini_net_set
          #   @synaptic_layer_indexes = calc_synaptic_layer_indexes
          #   @time_col_indexes = calc_time_col_indexes

          #   @synaptic_layer_indexes_reversed = @synaptic_layer_indexes.reverse
          #   @time_col_indexes_reversed = @time_col_indexes.reverse

          #   @synaptic_layer_index_last = @valid ? @synaptic_layer_indexes.last : -1
          #   @time_col_index_last = @valid ? @time_col_indexes.last : -1
          #   @node_output_sizes = calc_node_output_sizes

          #   # add/update/implement reverse time-col direction
          #   @node_input_sizes = calc_node_input_sizes

          #   @mini_net_set = init_mini_net_set

          #   @all_output_errors = synaptic_layer_indexes.map { time_col_indexes.map { 0.0 } }

          #   @input_set_given = Array(Array(Float64)).new
          #   @output_set_expected = Array(Array(Float64)).new
          # end

          # # This stays as-is:
          # def calc_node_input_sizes
          #   if @valid
          #     input_sizes = [input_size] + node_output_sizes[0..-2]
          #     synaptic_layer_indexes.map do |li|
          #       in_size = input_sizes[li]
          #       output_size = node_output_sizes[li]
          #       time_col_indexes.map do |ti|
          #         if ti == 0
          #           {previous_synaptic_layer: in_size, previous_time_column: 0}
          #         else
          #           {previous_synaptic_layer: in_size, previous_time_column: output_size}
          #         end
          #       end
          #     end
          #   else
          #     [[{previous_synaptic_layer: 0, previous_time_column: 0}]]
          #   end
          # end

          # # This gets added/implemented similar to 'calc_node_input_sizes', but in reverse time-col direction:
          # def calc_node_rev_input_sizes
          #   # TODO
          # end

          # # This stays as-is:
          # def init_mini_net_set
          #   synaptic_layer_indexes.map do |li|
          #     # NOTE: It should suffice to have bias only on the first li nets.
          #     #   So, force bias only on 1st and none on others
          #     li_gt_0 = li != 0

          #     mn_output_size = node_output_sizes[li]
          #     time_col_indexes.map do |ti|
          #       mn_input_size = node_input_sizes[li][ti].values.sum
          #       Cmn::MiniNet.new(
          #         height: mn_input_size,
          #         width: mn_output_size,

          #         learning_style: @learning_style,
          #         deriv_scale: @deriv_scale,

          #         bias_disabled: li_gt_0,
          #         bias_default: @bias_default,

          #         learning_rate: @learning_rate,
          #         momentum: @momentum,

          #         weight_init_scale: @weight_init_scale
          #       )
          #     end
          #   end
          # end

          # # This gets added/implemented similar to RnnSimple's 'init_mini_net_set', but with reverse time-col direction:
          # def init_mini_net_set
          #   # TODO: How to intertwine both the forward and the backward time-column paths?
          # end
        end
      end
    end
  end
end
