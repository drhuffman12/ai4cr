require "json"
require "./../learning_style.cr"

module Ai4cr
  module NeuralNetwork
    module Cmn
      module RnnConcerns
        module PropsAndInits
          getter config : RnnConcerns::NetConfig

          getter hidden_layer_index_max : Int32
          getter hidden_layer_range : Array(Int32)
          getter time_col_index_max : Int32
          getter time_col_range : Array(Int32)

          property weight_set_configs : Array(Array(RnnConcerns::WeightSetConfig))

          # property net_set : Array(Array(MiniNet)) # TODO

          def initialize(@config = RnnConcerns::NetConfig.new)
            @hidden_layer_index_max = @config.hidden_layer_qty - 1
            @hidden_layer_range = (0..@hidden_layer_index_max).to_a

            @time_col_index_max = @config.time_col_qty - 1
            @time_col_range = (0..@time_col_index_max).to_a

            @weight_set_configs = init_weight_set_configs

            # @net_set = init_net_set # TODO
          end

          def init_weight_set_configs
            @hidden_layer_range.map do |h|
              output_state_size = (h == @hidden_layer_index_max) ? @config.output_state_size : @config.hidden_state_size
              input_prev_layer_size = (h == 0) ? @config.input_state_size : @config.hidden_state_size
              hist_state_size = (h == @hidden_layer_index_max) ? @config.output_state_size : @config.hidden_state_size
              learing_style = case h
                              when 0
                                @config.hidden_learing_styles_first
                              when @hidden_layer_index_max
                                @config.output_learing_style
                              else
                                @config.hidden_learing_styles_middle
                              end

              # We only need to monitor error hist at final output
              error_distance_history_max = (h == 0) ? 0 : @config.error_distance_history_max

              @time_col_range.map do |t|
                hist_qty = (@config.hist_qty_max > t) ? @config.hist_qty_max - t : @config.hist_qty_max
                input_hist_set_sizes = (0..hist_qty - 1).to_a.map { hist_state_size }

                RnnConcerns::WeightSetConfig.new(
                  output_state_size: output_state_size,
                  input_prev_layer_size: input_prev_layer_size,
                  input_hist_set_sizes: input_hist_set_sizes,

                  # Bias, if any, is only at first level; not needed elsewhere
                  bias_disabled: (h == 0) ? @config.initial_bias_disabled : false,
                  bias_scale: (h == 0) ? @config.initial_bias_scale : 0.0,

                  learing_style: learing_style,
                  learning_rate: @config.learning_rate,
                  momentum: @config.momentum,
                  deriv_scale: @config.deriv_scale,
                  error_distance_history_max: error_distance_history_max,
                )
              end
            end
          end

          # def init_net_set # TODO
          # end

        end
      end
    end
  end
end
