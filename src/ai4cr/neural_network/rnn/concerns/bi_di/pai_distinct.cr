module Ai4cr
  module NeuralNetwork
    module Rnn
      module Concerns
        module BiDi
          module PaiDistinct
            # 'Pai' aka short for 'PropsAndInits'
            property node_input_sizes = NodeInputConfig.new
            property mini_net_set = MiniNetSet.new

            # getter input_sizes

            # ameba:disable Metrics/CyclomaticComplexity
            def calc_node_input_sizes : NodeInputConfig
              if @valid
                # @input_sizes = [input_size] + node_output_sizes[0..-2]
                input_sizes = [input_size] + node_output_sizes[0..-2]
                synaptic_layer_indexes.map do |sli|
                  # in_size = input_sizes[sli]
                  # output_size = node_output_sizes[sli]
                  time_col_indexes.map do |tci|
                    {
                      channel_forward: {
                        # enabled: sli > 0,
                        current_self_mem:            sli == 0 ? 0 : hidden_size,
                        sl_previous_input_or_combo:  sli == 0 ? 0 : input_sizes[sli],
                        sl_previous_channel_forward: sli <= 1 ? 0 : hidden_size,
                        tc_previous_channel_forward: sli == 0 ? 0 : (tci == 0 ? 0 : hidden_size),
                      },
                      channel_backward: {
                        # enabled: sli > 0,
                        current_self_mem:             sli == 0 ? 0 : hidden_size,
                        sl_previous_input_or_combo:   sli == 0 ? 0 : input_sizes[sli],
                        sl_previous_channel_backward: sli <= 1 ? 0 : hidden_size,
                        tc_next_channel_backward:     sli == 0 ? 0 : (tci == time_col_indexes.last ? 0 : hidden_size),
                      },
                      channel_input_or_combo: {
                        # disabled: false,
                        current_self_mem:           output_size = node_output_sizes[sli],
                        sl_previous_input_or_combo: input_sizes[sli],
                        current_forward:            sli == 0 ? 0 : hidden_size,
                        current_backward:           sli == 0 ? 0 : hidden_size,
                      },
                    }
                  end
                end
              else
                # for type consistency when otherwise invalid:
                NodeInputConfig.new
              end
            end

            # alias OutputDeltasFrom = NamedTuple(
            #   sli: Int32,
            #   tci: Int32,
            #   channel: Symbol,
            #   i_from: Int32,
            #   i_to: Int32
            # )
            # alias InputDeltasFor = Array(OutputDeltasFrom)
            # alias OutputDeltasFromInputDeltasFor = Array(
            #   Hash(Symbol,InputDeltasFor)
            # )

            # alias InputDeltasConfigsPerChannel = NamedTuple(
            #   channel_forward: NamedTuple(
            #     # enabled: Bool,
            #     current_self_mem: InputDeltasOffsets,
            #     sl_current_input_or_combo: InputDeltasOffsets,
            #     sl_next_channel_forward: InputDeltasOffsets,
            #     tc_next_channel_forward: InputDeltasOffsets
            #   ),
            #   channel_backward: NamedTuple(
            #     # enabled: Bool,
            #     current_self_mem: InputDeltasOffsets,
            #     sl_current_input_or_combo: InputDeltasOffsets,
            #     sl_next_channel_backward: InputDeltasOffsets,
            #     tc_previous_channel_backward: InputDeltasOffsets
            #   ),
            #   channel_input_or_combo: NamedTuple(
            #     current_self_mem: InputDeltasOffsets,
            #     sl_next_input_or_combo: InputDeltasOffsets,
            #     sl_next_forward: InputDeltasOffsets,
            #     sl_next_backward: InputDeltasOffsets
            #   )
            # )

            alias InputDeltasOffsets = NamedTuple(
              sli: Int32,
              tci: Int32,
              channel: Symbol,
              i_from: Int32,
              i_to: Int32)

            alias ChannelSymbol = Symbol
            alias IoPathSymbol = Symbol
            alias InputDeltasPerIoPath = Hash(IoPathSymbol,InputDeltasOffsets)
            alias InputDeltasConfigsPerChannel = Hash(ChannelSymbol, InputDeltasPerIoPath)

            alias OutputDeltasFromInputDeltasConfig = Array(Array(InputDeltasConfigsPerChannel))

            def output_deltas_from_input_deltas_for(sli, tci, channel) : OutputDeltasFromInputDeltasConfig
              if @valid
                synaptic_layer_indexes.map do |sli|
                  # in_size = input_sizes[sli]
                  # output_size = node_output_sizes[sli]
                  time_col_indexes.map do |tci|
                    hash_re_channel = InputDeltasConfigsPerChannel.new

                    if sli > 0
                      hash_re_channel.merge(odsfids_for_forward(sli, tci))
                      hash_re_channel.merge(odsfids_for_backward(sli, tci))
                    end
                    hash_re_channel.merge(odsfids_for_input_or_combo(sli, tci))

                    hash_re_channel
                  end
                end
              end
            end

            ################################################################
            # odsfids_for_forward*

            def odsfids_for_forward(sli, tci) : InputDeltasPerIoPath
              idpio = InputDeltasPerIoPath.new

              idpio
            end

            ################################################################
            # odsfids_for_backward*

            def odsfids_for_backward(sli, tci) : InputDeltasPerIoPath
              idpio = InputDeltasPerIoPath.new

              idpio
            end

            ################################################################
            # odsfids_for_input_or_combo*

            def odsfids_for_input_or_combo(sli, tci) : InputDeltasPerIoPath
              idpio = InputDeltasPerIoPath.new

              idpio[:current_self_mem] = odsfids_for_input_or_combo_re_current_self_mem(sli, tci)
              idpio[:sl_next_input_or_combo] = odsfids_for_input_or_combo_re_sl_next_input_or_combo(sli, tci)
              idpio[:sl_next_forward] = odsfids_for_input_or_combo_re_sl_next_forward(sli, tci)
              idpio[:sl_next_backward] = odsfids_for_input_or_combo_re_sl_next_backward(sli, tci)

              idpio
            end

            def odsfids_for_input_or_combo_re_current_self_mem(sli, tci) : InputDeltasOffsets
              # i_to_size = case sli
              # when synaptic_layer_indexes.last
              #   output_size
              # else
              #   hidden_size
              # end
              channel_from = :channel_input_or_combo
              i_to_size = node_input_sizes[sli][tci][channel_from][:current_self_mem]

              {
                sli:     sli,
                tci:     tci,
                channel: channel_from,
                i_from:  0,
                i_to:    i_to_size - 1,
              }
            end

            def odsfids_for_input_or_combo_re_sl_next_input_or_combo(sli, tci) : InputDeltasOffsets
              # raise "Bad sli index; sli: #{sli}" if sli == synaptic_layer_indexes.last
              return InputDeltasOffsets.new if sli == synaptic_layer_indexes[-1]

              sli_from = sli + 1
              tci_from = tci
              channel_from = :channel_input_or_combo

              i_from = node_input_sizes[sli_from][tci_from][channel_from][:current_self_mem]
              i_to_size = mini_net_set[sli][tci][:channel_input_or_combo].width
              i_to = i_from + i_to_size - 1

              {
                sli:     sli_from,
                tci:     tci_from,
                channel: channel_from,
                i_from:  i_from,
                i_to:    i_to,
              }
            end

            def odsfids_for_input_or_combo_re_sl_next_forward(sli, tci) : InputDeltasOffsets
              # raise "Bad sli index; sli: #{sli}" if sli == synaptic_layer_indexes.last
              return InputDeltasOffsets.new if sli == synaptic_layer_indexes[-1]

              sli_from = sli + 1
              tci_from = tci
              channel_from = :channel_forward

              i_from = node_input_sizes[sli_from][tci_from][channel_from][:current_self_mem]
              i_to_size = mini_net_set[sli][tci][:channel_input_or_combo].width
              i_to = i_from + i_to_size - 1

              {
                sli:     sli_from,
                tci:     tci_from,
                channel: channel_from,
                i_from:  i_from,
                i_to:    i_to,
              }
            end

            def odsfids_for_input_or_combo_re_sl_next_backward(sli, tci) : InputDeltasOffsets
              # raise "Bad sli index; sli: #{sli}" if sli == synaptic_layer_indexes.last
              return InputDeltasOffsets.new if sli == synaptic_layer_indexes[-1]

              sli_from = sli + 1
              tci_from = tci
              channel_from = :channel_backward

              i_from = node_input_sizes[sli_from][tci_from][channel_from][:current_self_mem]
              i_to_size = mini_net_set[sli][tci][:channel_input_or_combo].width
              i_to = i_from + i_to_size - 1

              {
                sli:     sli_from,
                tci:     tci_from,
                channel: channel_from,
                i_from:  i_from,
                i_to:    i_to,
              }
            end

            # def output_deltas_from_input_deltas_for(sli, tci, channel) : InputDeltasConfig
            #   if @valid
            #     synaptic_layer_indexes.map do |sli|
            #       # in_size = input_sizes[sli]
            #       # output_size = node_output_sizes[sli]
            #       time_col_indexes.map do |tci|
            #         hash = OutputDeltasFromInputDeltasFor.new

            #         if sli > 0
            #           hash.merge(odsfids_for_forward(sli, tci))
            #           hash.merge(odsfids_for_backward(sli, tci))
            #         end
            #         hash.merge(odsfids_for_input_or_combo(sli, tci))

            #         hash
            #       end
            #     end
            #   end
            # end

            # def odsfids_for_input_or_combo(sli, tci)
            #   channel = :channel_input_or_combo
            #   arr = InputDeltasFor.new

            #   next_entry = odsfids_for_input_or_combo_re_current_self_mem(sli, tci)
            #   arr << next_entry

            #   if sli < synaptic_layer_indexes.last
            #     # next_input_or_combo
            #     i_from = next_entry.values[:i_to] + 1

            #     next_entry = odsfids_for_input_or_combo_re_sl_next_input_or_combo(sli, tci, i_from)
            #     arr << next_entry
            #   end

            #   { channel => arr }
            # end

            # def odsfids_for_input_or_combo_re_sl_next_input_or_combo(sli, tci, i_from)
            #   i_to_size = case sli
            #   when 0
            #     input_size
            #   when synaptic_layer_indexes.last
            #     0
            #   else
            #     hidden_size
            #   end

            #   i_to = i_from + i_to_size -1

            #   {
            #     # i_to_size: i_to_size,
            #     # i_to: i_to
            #     mn_from: {
            #       sli: sli + 1,
            #       tci: tci,
            #       channel: channel,
            #       i_from: i_from,
            #       i_to: i_to
            #     }
            #   }
            # end

            # ameba:enable Metrics/CyclomaticComplexity

            def init_mini_net_set
              sli_last = synaptic_layer_indexes.last
              synaptic_layer_indexes.map do |sli|
                # NOTE: It should suffice to have bias only on the first sli nets.
                #   So, force bias only on 1st and none on others
                sli_gt_0 = sli != 0

                # Alternate thru the sequence of learning styles
                lsi = sli % @learning_styles.size

                # mn_output_size = node_output_sizes[sli]
                time_col_indexes.map do |tci|
                  h = Hash(Symbol, Ai4cr::NeuralNetwork::Cmn::MiniNet).new
                  if sli == 0 # && channel_symbol != :channel_input_or_combo
                    # ONLY add mini_net for channel :channel_input_or_combo
                    h[:channel_input_or_combo] = gen_mini_net_4_channel_input_or_combo(sli, tci, sli_gt_0, lsi, sli_last)
                  else
                    # Add mini_net for all channels
                    h[:channel_forward] = gen_mini_net_4_channel_forward(sli, tci, sli_gt_0, lsi)
                    h[:channel_backward] = gen_mini_net_4_channel_backward(sli, tci, sli_gt_0, lsi)
                    h[:channel_input_or_combo] = gen_mini_net_4_channel_input_or_combo(sli, tci, sli_gt_0, lsi, sli_last)
                  end

                  h
                end
              end
            end

            def gen_mini_net_4_channel_forward(sli, tci, sli_gt_0, lsi)
              mn_input_size = node_input_sizes[sli][tci][:channel_forward].values.sum
              mn_output_size = hidden_size
              gen_mini_net(sli, sli_gt_0, lsi, mn_input_size, mn_output_size)
            end

            def gen_mini_net_4_channel_backward(sli, tci, sli_gt_0, lsi)
              mn_input_size = node_input_sizes[sli][tci][:channel_backward].values.sum
              mn_output_size = hidden_size
              gen_mini_net(sli, sli_gt_0, lsi, mn_input_size, mn_output_size)
            end

            def gen_mini_net_4_channel_input_or_combo(sli, tci, sli_gt_0, lsi, sli_last)
              mn_input_size = node_input_sizes[sli][tci][:channel_input_or_combo].values.sum
              mn_output_size = (sli == sli_last) ? output_size : hidden_size
              gen_mini_net(sli, sli_gt_0, lsi, mn_input_size, mn_output_size)
            end

            def gen_mini_net(sli, sli_gt_0, lsi, mn_input_size, mn_output_size)
              Cmn::MiniNet.new(
                height: mn_input_size,
                width: mn_output_size,

                learning_styles: @learning_styles[lsi],
                deriv_scale: @deriv_scale,

                bias_disabled: sli_gt_0,
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
