require "json"
require "./../learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module PropsAndInits
          getter config : RnnConcerns::NetConfig

          getter layer_index_max : Int32
          getter layer_range : Array(Int32)
          getter time_col_index_max : Int32
          getter time_col_range : Array(Int32)

          property mini_net_configs : Array(Array(RnnConcerns::MiniNetConfig))

          property mini_net_set : Array(Array(MiniNet)) # TODO

          def initialize(@config = RnnConcerns::NetConfig.new)
            @layer_index_max = @config.hidden_layer_qty
            @layer_range = (0..@layer_index_max).to_a

            @time_col_index_max = @config.time_col_qty - 1
            @time_col_range = (0..@time_col_index_max).to_a

            @mini_net_configs = init_mini_net_configs

            @mini_net_set = init_mini_net_set # TODO
          end

          def init_mini_net_configs
            @layer_range.map do |h|
              output_state_size = (h == @layer_index_max) ? @config.output_state_size : @config.hidden_state_size
              input_prev_layer_size = (h == 0) ? @config.input_state_size : @config.hidden_state_size
              hist_state_size = (h == @layer_index_max) ? @config.output_state_size : @config.hidden_state_size

              bias_disabled = (h == 0) ? @config.initial_bias_disabled : true
              bias_scale = (h == 0) ? @config.initial_bias_scale : 0.0

              learning_style = case h
                               when 0
                                 @config.hidden_learning_styles_first
                               when @layer_index_max
                                 @config.output_learning_style
                               else
                                 @config.hidden_learning_styles_middle
                               end

              # We only need to monitor error hist at final output
              error_distance_history_max = (h == 0) ? 0 : @config.error_distance_history_max

              @time_col_range.map do |t|
                hist_qty = (@config.hist_qty_max > t) ? [t - @config.hist_qty_max + 1, @config.hist_qty_max].min : 0
                input_hist_set_sizes = (0..hist_qty - 1).to_a.map { hist_state_size }

                RnnConcerns::MiniNetConfig.new(
                  output_state_size: output_state_size,
                  input_prev_layer_size: input_prev_layer_size,
                  input_hist_set_sizes: input_hist_set_sizes,

                  # Bias, if any, is only at first level; not needed elsewhere
                  bias_disabled: bias_disabled,
                  bias_scale: bias_scale,

                  learning_style: learning_style,
                  learning_rate: @config.learning_rate,
                  momentum: @config.momentum,
                  deriv_scale: @config.deriv_scale,
                  error_distance_history_max: error_distance_history_max,
                )
              end
            end
          end

          def init_mini_net_set
            @layer_range.map do |h|
              @time_col_range.map do |t|
                Cmn::MiniNet.new(mini_net_configs[h][t])
              end
            end
          end
        end
      end
    end
  end
end
