module Ai4cr
  module NeuralNetwork
    module Rnn
      module RnnSimpleConcerns
        module PropsAndInits
          HISTORY_SIZE_DEFAULT    = 10
          IO_OFFSET_DEFAULT       =  1
          TIME_COL_QTY_MIN        =  2
          HIDDEN_LAYER_QTY_MIN    =  1
          INPUT_SIZE_MIN          =  2 # 1 # TODO: Could be just '1', but will need to adjust a bunch of tests!
          OUTPUT_SIZE_MIN         =  1
          LEARNINGS_STYLE_DEFAULT = [LS_RELU, LS_SIGMOID]

          # TODO: Handle usage of a 'structure' param in 'initialize'
          # def initialize(@time_col_qty = TIME_COL_QTY_MIN, @structure = [INPUT_SIZE_MIN, OUTPUT_SIZE_MIN])
          #   initialize(time_col_qty, structure[0], structure[-1], structure[1..-2], )
          # end

          def config
            {
              name: name,

              history_size: history_size,

              io_offset:         @io_offset,
              time_col_qty:      @time_col_qty,
              input_size:        @input_size,
              output_size:       @output_size,
              hidden_layer_qty:  @hidden_layer_qty,
              hidden_size_given: @hidden_size_given,

              learning_styles: @learning_styles,

              bias_disabled: @bias_disabled,

              bias_default: @bias_default,

              learning_rate: @learning_rate,
              momentum:      @momentum,
              deriv_scale:   @deriv_scale,

              weight_init_scale_given: @weight_init_scale,
            }
          end

          def initialize(
            name : String? = "",

            history_size : Int32? = HISTORY_SIZE_DEFAULT,
            @io_offset = IO_OFFSET_DEFAULT,
            @time_col_qty = TIME_COL_QTY_MIN,
            @input_size = INPUT_SIZE_MIN,
            @output_size = OUTPUT_SIZE_MIN,
            @hidden_layer_qty = HIDDEN_LAYER_QTY_MIN,
            @hidden_size_given = 0,

            @learning_styles : Array(LearningStyle) = LEARNINGS_STYLE_DEFAULT,

            bias_disabled = false,

            bias_default = 1.0,

            learning_rate : Float64? = nil,
            momentum : Float64? = nil,
            deriv_scale : Float64? = nil,

            weight_init_scale_given : Float64? = nil
          )
            @name = name.nil? ? "" : name

            init_network(hidden_size_given, bias_disabled, bias_default, learning_rate, momentum, deriv_scale)

            #             @weight_init_scale = case
            #                                  when weight_init_scale_given.nil?
            #                                    ![LS_PRELU, LS_RELU].includes?(learning_styles.first) ? 1.0 : 1.0 / ( #  (time_col_qty ** 2) * (input_size ** 2) * (hidden_layer_qty ** 2) * (hidden_size ** 2) * (output_size ** 2) # * 100 # **2
            # (time_col_qty * input_size * hidden_layer_qty * hidden_size * output_size) * 1000                 # * time_col_qty
            # )
            #                                  else
            #                                    weight_init_scale_given
            #                                  end
            @weight_init_scale = 1.0 / (time_col_qty * input_size * hidden_layer_qty * hidden_size * output_size * 1000)

            @error_stats = Ai4cr::ErrorStats.new(history_size)
          end

          def init_network(
            hidden_size_given,
            bias_disabled,
            bias_default,
            learning_rate,
            momentum,
            deriv_scale
          )
            init_network_config(hidden_size_given, bias_disabled, bias_default, learning_rate, momentum, deriv_scale)
            init_network_mini_net_set
          end

          def init_network_config(
            hidden_size_given,
            bias_disabled,
            bias_default,
            learning_rate,
            momentum,
            deriv_scale
          )
            # TODO: Handle differing hidden layer output sizes
            if hidden_size_given > 0
              @hidden_size = hidden_size_given
            else
              @hidden_size = @input_size + @output_size
            end

            # TODO: switch 'disabled_bias' to 'enabled_bias' and adjust defaulting accordingly
            @bias_disabled = bias_disabled
            @bias_default = bias_default

            @learning_rate = learning_rate.nil? || learning_rate.as(Float64) <= 0.0 ? Ai4cr::Utils::Rand.rand_excluding : learning_rate.as(Float64)
            @momentum = momentum.nil? || momentum.as(Float64) <= 0.0 ? Ai4cr::Utils::Rand.rand_excluding : momentum.as(Float64)
            @deriv_scale = deriv_scale.nil? || deriv_scale.as(Float64) <= 0.0 ? Ai4cr::Utils::Rand.rand_excluding / 2.0 : deriv_scale.as(Float64)

            @valid = false
            @errors = Hash(String, String).new
            validate!

            @synaptic_layer_qty = hidden_layer_qty + 1
          end

          def init_network_mini_net_set
            @synaptic_layer_indexes = calc_synaptic_layer_indexes
            @time_col_indexes = calc_time_col_indexes

            @synaptic_layer_indexes_reversed = @synaptic_layer_indexes.reverse
            @time_col_indexes_reversed = @time_col_indexes.reverse

            @synaptic_layer_index_last = @valid ? @synaptic_layer_indexes.last : -1
            @time_col_index_last = @valid ? @time_col_indexes.last : -1
            @node_output_sizes = calc_node_output_sizes
            @node_input_sizes = calc_node_input_sizes

            @mini_net_set = init_mini_net_set

            @all_output_errors = synaptic_layer_indexes.map { time_col_indexes.map { 0.0 } }

            @input_set_given = Array(Array(Float64)).new
            @output_set_expected = Array(Array(Float64)).new
          end

          def valid?
            @valid
          end

          def validate!
            @errors = Hash(String, String).new

            @errors["time_col_qty"] = "time_col_qty must be at least #{TIME_COL_QTY_MIN}!" if time_col_qty < TIME_COL_QTY_MIN
            @errors["hidden_layer_qty"] = "hidden_layer_qty must be at least #{HIDDEN_LAYER_QTY_MIN}!" if hidden_layer_qty < HIDDEN_LAYER_QTY_MIN
            @errors["input_size"] = "input_size must be at least #{INPUT_SIZE_MIN}" if input_size < INPUT_SIZE_MIN
            @errors["output_size"] = "output_size must be at least #{OUTPUT_SIZE_MIN}" if output_size < OUTPUT_SIZE_MIN
            @errors["hidden_size_given"] = "hidden_size_given must NOT be negative" if !(!hidden_size_given.nil? && hidden_size_given.to_i > -1)
            @errors["io_offset"] = "io_offset must be a non-negative integer" if io_offset < 0

            @valid = errors.empty?

            raise errors.to_json unless @valid

            @valid
          end

          def calc_synaptic_layer_indexes
            if @valid
              Array.new(@synaptic_layer_qty) { |i| i }
            else
              [] of Int32
            end
          end

          def calc_time_col_indexes
            if @valid
              Array.new(@time_col_qty) { |i| i }
            else
              [] of Int32
            end
          end

          def calc_node_output_sizes
            if @valid
              synaptic_layer_indexes.map do |li|
                li == synaptic_layer_index_last ? output_size : hidden_size
              end
            else
              [] of Int32
            end
          end

          def calc_node_input_sizes
            if @valid
              input_sizes = [input_size] + node_output_sizes[0..-2]
              synaptic_layer_indexes.map do |li|
                in_size = input_sizes[li]
                output_size = node_output_sizes[li]
                time_col_indexes.map do |ti|
                  if ti == 0
                    {previous_synaptic_layer: in_size, previous_time_column: 0}
                  else
                    {previous_synaptic_layer: in_size, previous_time_column: output_size}
                  end
                end
              end
            else
              [[{previous_synaptic_layer: 0, previous_time_column: 0}]]
            end
          end

          def init_mini_net_set
            synaptic_layer_indexes.map do |li|
              # NOTE: It should suffice to have bias only on the first li nets.
              #   So, force bias only on 1st and none on others
              li_gt_0 = li != 0

              mn_output_size = node_output_sizes[li]
              time_col_indexes.map do |ti|
                mn_input_size = node_input_sizes[li][ti].values.sum

                # Alternate thru the sequence of learning styles
                lsi = li % @learning_styles.size

                Cmn::MiniNet.new(
                  height: mn_input_size,
                  width: mn_output_size,

                  learning_style: @learning_styles[lsi],
                  deriv_scale: @deriv_scale,

                  bias_disabled: li_gt_0,
                  bias_default: @bias_default,

                  learning_rate: @learning_rate,
                  momentum: @momentum,

                  weight_init_scale: @weight_init_scale
                )
              end
            end
          end
        end
      end
    end
  end
end
