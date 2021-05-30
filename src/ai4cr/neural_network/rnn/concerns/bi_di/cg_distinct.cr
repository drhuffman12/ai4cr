module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module CgDistinct
            CHANNELS_FIRST_SLI = [:channel_input_or_combo]
            CHANNELS_OTHER_SLI = [:channel_forward, :channel_backward, :channel_input_or_combo]

            # steps for 'eval' aka 'guess':
            def eval(input_set_given)
              # TODO: Review/compare w/ 'train' and adjust as applicable!

              @input_set_given = input_set_given

              # time_col_indexes_last = @time_col_indexes.last
              synaptic_layer_indexes.each do |sli|
                if sli > 0
                  channel = :channel_forward
                  time_col_indexes.each do |tci|
                    load_inputs_and_step_forward(sli, tci, channel)
                  end

                  channel = :channel_backward
                  time_col_indexes_reversed.each do |tci|
                    load_inputs_and_step_forward(sli, tci, channel)
                  end
                end

                channel = :channel_input_or_combo
                time_col_indexes.each do |tci|
                  load_inputs_and_step_forward(sli, tci, channel)
                end
              end

              outputs_guessed
            rescue ex
              msg = {
                my_msg:    "BROKE HERE!",
                file:      __FILE__,
                line:      __LINE__,
                klass:     ex.class,
                message:   ex.message,
                backtrace: ex.backtrace,
              }
              raise msg.to_s
            end

            def load_inputs_and_step_forward(sli, tci, channel)
              ins = inputs_for(sli, tci, channel).values.flatten
              mini_net_set[sli][tci][channel].step_load_inputs(ins)
              mini_net_set[sli][tci][channel].step_calc_forward
            end

            # ameba:disable Metrics/CyclomaticComplexity
            def inputs_for(sli, tci, channel)
              # ins = Array(Array(Float64)).new
              # ins = Array(Array(Hash(Symbol, Float64))).new
              ins = Hash(Symbol, Array(Float64)).new

              # memory
              if sli == 0 && channel == :channel_input_or_combo
                ins[:current_self_mem] = mini_net_set[sli][tci][channel].outputs_guessed
              elsif sli > 0
                ins[:current_self_mem] = mini_net_set[sli][tci][channel].outputs_guessed
              end

              # prior sli outputs (or original inputs if sli == 0)
              if sli == 0 && channel == :channel_input_or_combo
                ins[:sl_previous_input_or_combo] = @input_set_given[tci] # if channel == :channel_input_or_combo
              else
                ins[:sl_previous_input_or_combo] = mini_net_set[sli - 1][tci][:channel_input_or_combo].outputs_guessed
              end

              # current forward and backward into current combo
              if channel == :channel_input_or_combo
                if sli > 0
                  ins[:current_forward] = mini_net_set[sli][tci][:channel_forward].outputs_guessed
                  ins[:current_backward] = mini_net_set[sli][tci][:channel_backward].outputs_guessed
                end
              end

              # if channel == :channel_input_or_combo
              # ins << mini_net_set[sli][tci+1].outputs_guessed unless sli == 0
              # end

              if channel == :channel_forward
                # prior tci outputs (unless sli == 0 or tci == 0)
                ins[:sl_previous_channel_forward] = mini_net_set[sli - 1][tci][channel].outputs_guessed if sli > 1
                ins[:tc_previous_channel_forward] = mini_net_set[sli][tci - 1][channel].outputs_guessed unless sli == 0 || tci == 0
              end

              if channel == :channel_backward
                # next tci outputs (unless sli == 0 or tci == max tci)
                ins[:sl_previous_channel_backward] = mini_net_set[sli - 1][tci][channel].outputs_guessed if sli > 1
                ins[:tc_next_channel_backward] = mini_net_set[sli][tci + 1][channel].outputs_guessed unless sli == 0 || tci >= @time_col_indexes_last
              end

              # bias
              if sli == 0 && channel == :channel_input_or_combo
                # ins << mini_net_set[sli][tci][channel].outputs_guessed
                ins[:bias] = [@bias_default] if !@bias_disabled
                # elsif sli > 0
                #   # ins << mini_net_set[sli][tci][channel].outputs_guessed
                #   ins << [@bias_default] if !@bias_disabled
              end

              ins
            rescue ex
              p! ["vvvv", :inputs_for, sli, tci, channel]
              p! ex.class
              p! ex.message
              p! ex.backtrace.pretty_inspect
              p! ["^^^^", :inputs_for, sli, tci, channel]
              raise ex
            end

            # ameba:enable Metrics/CyclomaticComplexity

            # def all_mini_nets_each(&block)
            #   synaptic_layer_indexes.each do |sli|
            #     channels = (sli == 0) ? CHANNELS_FIRST_SLI : CHANNELS_OTHER_SLI
            #     time_col_indexes.each do |tci|
            #       channels.each do |channel|
            #         yield @mini_net_set[sli][tci][channel]
            #       end
            #     end
            #   end
            # end

            def all_mini_nets_each(&block)
              synaptic_layer_indexes.each do |sli|
                # channels = (sli == 0) ? CHANNELS_FIRST_SLI : CHANNELS_OTHER_SLI
                if sli == 0
                  time_col_indexes.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][:channel_input_or_combo]
                    # end
                  end
                else
                  channel = :channel_forward
                  time_col_indexes.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel]
                    # end
                  end

                  channel = :channel_backward
                  time_col_indexes_reversed.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel]
                    # end
                  end

                  channel = :channel_input_or_combo
                  time_col_indexes.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel]
                    # end
                  end
                end
              end
            end

            def all_mini_nets_each_with_indexes(&block)
              synaptic_layer_indexes.each do |sli|
                # channels = (sli == 0) ? CHANNELS_FIRST_SLI : CHANNELS_OTHER_SLI
                if sli == 0
                  time_col_indexes.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][:channel_input_or_combo], sli, tci, channel
                    # end
                  end
                else
                  channel = :channel_forward
                  time_col_indexes.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel], sli, tci, channel
                    # end
                  end

                  channel = :channel_backward
                  time_col_indexes_reversed.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel], sli, tci, channel
                    # end
                  end

                  channel = :channel_input_or_combo
                  time_col_indexes.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel], sli, tci, channel
                    # end
                  end
                end
              end
            end

            def all_mini_nets_each_reversed_with_indexes(&block)
              synaptic_layer_indexes_reversed.each do |sli|
                # channels = (sli == 0) ? CHANNELS_FIRST_SLI : CHANNELS_OTHER_SLI
                if sli == 0
                  time_col_indexes_reversed.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][:channel_input_or_combo], sli, tci, channel
                    # end
                  end
                else
                  channel = :channel_input_or_combo
                  time_col_indexes_reversed.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel], sli, tci, channel
                    # end
                  end

                  channel = :channel_backward
                  time_col_indexes.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel], sli, tci, channel
                    # end
                  end

                  channel = :channel_forward
                  time_col_indexes_reversed.each do |tci|
                    # channels.each do |channel|
                    yield @mini_net_set[sli][tci][channel], sli, tci, channel
                    # end
                  end
                end
              end
            end

            def all_mini_nets_map(&block)
              synaptic_layer_indexes.map do |sli|
                channels = (sli == 0) ? CHANNELS_FIRST_SLI : CHANNELS_OTHER_SLI
                time_col_indexes.map do |tci|
                  channels.map do |channel|
                    [channel, yield @mini_net_set[sli][tci][channel]]
                  end.to_h
                end
              end
            end

            def all_mini_nets_map_with_indexes(&block)
              synaptic_layer_indexes.map do |sli|
                channels = (sli == 0) ? CHANNELS_FIRST_SLI : CHANNELS_OTHER_SLI
                time_col_indexes.map do |tci|
                  channels.map do |channel|
                    [channel, yield @mini_net_set[sli][tci][channel], sli, tci, channel]
                  end.to_h
                end
              end
            end

            def map_only_indexes(&block)
              synaptic_layer_indexes.map do |sli|
                channels = (sli == 0) ? CHANNELS_FIRST_SLI : CHANNELS_OTHER_SLI
                time_col_indexes.map do |tci|
                  channels.map do |channel|
                    [channel, yield sli, tci, channel]
                  end.to_h
                end
              end
            end

            def weights
              all_mini_nets_map do |mini_net|
                mini_net.weights
              end # as(Array(Array(Hash(Symbol, Array(Array(Float64))))))
            end

            def weights=(w : BiDi::Weights)
              synaptic_layer_indexes.map do |sli|
                time_col_indexes.map do |tci|
                  if sli > 0
                    channel = :channel_forward
                    mini_net_set[sli][tci][channel].weights = w[sli][tci][channel]

                    channel = :channel_backward
                    mini_net_set[sli][tci][channel].weights = w[sli][tci][channel]
                  end

                  channel = :channel_input_or_combo
                  mini_net_set[sli][tci][channel].weights = w[sli][tci][channel]
                end
              end
            end

            def outputs_guessed
              sli = synaptic_layer_indexes.last

              time_col_indexes.map do |tci|
                a = mini_net_set[sli]
                b = a[tci]
                guessed = b[:channel_input_or_combo].outputs_guessed
                guessed
              end
            end

            # guesses
            def guesses_sorted
              sli = synaptic_layer_indexes.last

              time_col_indexes.map do |tci|
                mini_net_set[sli][tci].guesses_sorted
              end
            end

            def guesses_ceiled
              sli = synaptic_layer_indexes.last

              time_col_indexes.map do |tci|
                mini_net_set[sli][tci].guesses_ceiled
              end
            end

            def guesses_top_n(n)
              sli = synaptic_layer_indexes.last

              time_col_indexes.map do |tci|
                mini_net_set[sli][tci].guesses_top_n(n)
              end
            end
          end
        end
      end
    end
  end
end
