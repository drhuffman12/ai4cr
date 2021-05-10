module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module CgDistinct
            # TODO: Adapt below based on 'src/ai4cr/neural_network/rnn/concerns/bi_di/pai_distinct.cr'

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

                channel = :channel_sl_or_combo
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
              ins = inputs_for(sli, tci, channel).flatten
              mini_net_set[sli][tci][channel].step_load_inputs(ins)
              mini_net_set[sli][tci][channel].step_calc_forward
            end

            # def step_load_inputs_channel_sl_or_combo(sli, tci)
            #   mini_net_set[sli][tci].step_load_inputs(inputs_for(sli, tci, channel))
            # end

            # def step_load_inputs_channel_forward(sli, tci)
            # end

            # def step_load_inputs_channel_backward(sli, tci)
            # end

            # inputs_for

            # def inputs_for(sli, tci)
            #   case
            #   when sli == 0 && tci == 0
            #     @input_set_given[tci]
            #   when sli == 0 && tci > 0
            #     @input_set_given[tci] + step_outputs_guessed_from_previous_tc(sli, tci)
            #   when sli > 0 && tci == 0
            #     step_outputs_guessed_from_previous_li(sli, tci)
            #   else
            #     step_outputs_guessed_from_previous_li(sli, tci) + step_outputs_guessed_from_previous_tc(sli, tci)
            #   end
            # end

            # def inputs_for_channel_sl_or_combo
            # ameba:disable Metrics/CyclomaticComplexity
            def inputs_for(sli, tci, channel)
              # ins = Array(Array(Float64)).new
              # ins = Array(Array(Hash(Symbol, Float64))).new
              ins = Hash(Symbol, Array(Float64)).new

              # memory
              if sli == 0 && channel == :channel_sl_or_combo
                ins[:memory] = mini_net_set[sli][tci][channel].outputs_guessed
              elsif sli > 0
                ins[:memory] = mini_net_set[sli][tci][channel].outputs_guessed
              end

              # prior sli outputs (or original inputs if sli == 0)
              if sli == 0 && channel == :channel_sl_or_combo
                ins[:prior_sli] = @input_set_given[tci] # if channel == :channel_sl_or_combo
              else
                ins[:prior_sli] = mini_net_set[sli - 1][tci][:channel_sl_or_combo].outputs_guessed
              end

              # current forward and backward into current combo
              if channel == :channel_sl_or_combo
                if sli > 0
                  ins[:current_forward] = mini_net_set[sli][tci][:channel_forward].outputs_guessed
                  ins[:current_backward] = mini_net_set[sli][tci][:channel_backward].outputs_guessed
                end
              end

              # if channel == :channel_sl_or_combo
              # ins << mini_net_set[sli][tci+1].outputs_guessed unless sli == 0
              # end

              if channel == :channel_forward
                # prior tci outputs (unless sli == 0 or tci == 0)
                ins[:prior_forward] = mini_net_set[sli][tci - 1][channel].outputs_guessed unless sli == 0 || tci == 0
              end

              if channel == :channel_backward
                # next tci outputs (unless sli == 0 or tci == max tci)
                ins[:prior_backward] = mini_net_set[sli][tci + 1][channel].outputs_guessed unless sli == 0 || tci >= @time_col_indexes_last
              end

              # bias
              if sli == 0 && channel == :channel_sl_or_combo
                # ins << mini_net_set[sli][tci][channel].outputs_guessed
                ins[:bias] = [@bias_default] if !@bias_disabled
                # elsif sli > 0
                #   # ins << mini_net_set[sli][tci][channel].outputs_guessed
                #   ins << [@bias_default] if !@bias_disabled
              end

              # if sli == 0 && channel == :channel_sl_or_combo
              #   ins << [@bias_default] if !@bias_disabled
              # # else
              # #   ins << [] of Float64
              # end

              ins # .flatten

              # if sli == 0
              #   input_set_given[tci]
              # else
              #   mini_net_set[sli-1][tci].outputs_guessed
              # end

              # case
              # when sli == 0 && tci == 0
              #   @input_set_given[tci]
              # when sli == 0 && tci > 0
              #   @input_set_given[tci] + step_outputs_guessed_from_previous_tc(sli, tci)
              # when sli > 0 && tci == 0
              #   step_outputs_guessed_from_previous_li(sli, tci)
              # else
              #   step_outputs_guessed_from_previous_li(sli, tci) + step_outputs_guessed_from_previous_tc(sli, tci)
              # end


            rescue ex
              p! ["vvvv", :inputs_for, sli, tci, channel]
              p! ex.class
              p! ex.message
              p! ex.backtrace.pretty_inspect
              p! ["^^^^", :inputs_for, sli, tci, channel]
              raise ex
            end

            # ameba:enable Metrics/CyclomaticComplexity

            def outputs_guessed
              sli = synaptic_layer_indexes.last

              time_col_indexes.map do |tci|
                a = mini_net_set[sli]
                b = a[tci]
                guessed = b.outputs_guessed
                guessed
              end
            end

            # private def step_outputs_guessed_from_previous_tc(sli, tci)
            #   raise "Index error in step_outputs_guessed_from_previous_tc" if tci == 0

            #   mini_net_set[sli][tci - 1].outputs_guessed
            # end

            # private def step_outputs_guessed_from_previous_li(sli, tci)
            #   raise "Index error in step_outputs_guessed_from_previous_li" if sli == 0

            #   mini_net_set[sli - 1][tci].outputs_guessed
            # end

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
