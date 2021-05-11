require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::CgDistinct do
  describe "#inputs_for" do
    let(hidden_size_given) { 0 } # aka the default
    let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

    let(input_set_given_example) {
      [
        [0.0, 0.0, 0.0],
        [0.0, 0.0],
      ]
    }
    let(input_set_given_example) { [[0.1, 0.2], [0.3, 0.4]] }

    before_each do
      rnn_bi_di.input_set_given = input_set_given_example
    end

    let(node_input_sizes_expected) {
      [
        # sli: 0
        [
          # sli: 0, tci: 0
          {
            channel_forward: {
              current_self_mem:            0,
              sl_previous_input_or_combo:  0,
              sl_previous_channel_forward: 0,
              tc_previous_channel_forward: 0,
            },
            channel_backward: {
              current_self_mem:             0,
              sl_previous_input_or_combo:   0,
              sl_previous_channel_backward: 0,
              tc_next_channel_backward:     0,
            },
            channel_sl_or_combo: {
              current_self_mem:           rnn_bi_di.hidden_size,
              sl_previous_input_or_combo: rnn_bi_di.input_size,
              current_forward:            0,
              current_backward:           0,
            },
          },

          # sli: 0, tci: 1
          {
            channel_forward: {
              current_self_mem:            0,
              sl_previous_input_or_combo:  0,
              sl_previous_channel_forward: 0,
              tc_previous_channel_forward: 0,
            },
            channel_backward: {
              current_self_mem:             0,
              sl_previous_input_or_combo:   0,
              sl_previous_channel_backward: 0,
              tc_next_channel_backward:     0,
            },
            channel_sl_or_combo: {
              current_self_mem:           rnn_bi_di.hidden_size,
              sl_previous_input_or_combo: rnn_bi_di.input_size,
              current_forward:            0,
              current_backward:           0,
            },
          },
        ],

        # sli: 1
        [
          # sli: 1, tci: 0
          {
            channel_forward: {
              current_self_mem:            rnn_bi_di.hidden_size,
              sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
              sl_previous_channel_forward: 0,
              tc_previous_channel_forward: 0,
            },
            channel_backward: {
              current_self_mem:             rnn_bi_di.hidden_size,
              sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
              sl_previous_channel_backward: 0,
              tc_next_channel_backward:     rnn_bi_di.hidden_size,
            },
            channel_sl_or_combo: {
              current_self_mem:           rnn_bi_di.output_size,
              sl_previous_input_or_combo: rnn_bi_di.hidden_size,
              current_forward:            rnn_bi_di.hidden_size,
              current_backward:           rnn_bi_di.hidden_size,
            },
          },

          # sli: 1, tci: 1
          {
            channel_forward: {
              current_self_mem:            rnn_bi_di.hidden_size,
              sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
              sl_previous_channel_forward: 0,
              tc_previous_channel_forward: rnn_bi_di.hidden_size,
            },
            channel_backward: {
              current_self_mem:             rnn_bi_di.hidden_size,
              sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
              sl_previous_channel_backward: 0,
              tc_next_channel_backward:     0,
            },
            channel_sl_or_combo: {
              current_self_mem:           rnn_bi_di.output_size,
              sl_previous_input_or_combo: rnn_bi_di.hidden_size,
              current_forward:            rnn_bi_di.hidden_size,
              current_backward:           rnn_bi_di.hidden_size,
            },
          },
        ],
      ]
    }

    let(node_input_sets_expected) {
    }

    def expect_input_sizes_channel_sl_or_combo(rnn_bi_di, node_input_sizes_expected)
      rnn_bi_di.synaptic_layer_indexes.each do |sli|
        include_bias = sli == 0
        rnn_bi_di.time_col_indexes.each do |tci|
          channel = :channel_sl_or_combo
          expect_input_sizes(rnn_bi_di, node_input_sizes_expected, sli, tci, channel, include_bias)
        end
      end
    end

    def expect_input_sizes_channel_forward(rnn_bi_di, node_input_sizes_expected)
      rnn_bi_di.synaptic_layer_indexes.each do |sli|
        rnn_bi_di.time_col_indexes.each do |tci|
          if sli != 0
            channel = :channel_forward
            expect_input_sizes(rnn_bi_di, node_input_sizes_expected, sli, tci, channel)
          end
        end
      end
    end

    def expect_input_sizes_channel_backward(rnn_bi_di, node_input_sizes_expected)
      rnn_bi_di.synaptic_layer_indexes.each do |sli|
        rnn_bi_di.time_col_indexes.each do |tci|
          if sli != 0
            channel = :channel_backward
            expect_input_sizes(rnn_bi_di, node_input_sizes_expected, sli, tci, channel)
          end
        end
      end
    end

    def expect_input_sizes(rnn_bi_di, node_input_sizes_expected, sli, tci, channel, include_bias = false)
      input_size_expected = (include_bias ? 1 : 0) + node_input_sizes_expected[sli][tci][channel].values.sum
      inputs_for = rnn_bi_di.inputs_for(sli, tci, channel)
      input_sizes_actual = inputs_for.values.flatten.size

      puts
      p! sli
      p! tci
      p! channel
      p! node_input_sizes_expected[sli][tci][channel]
      p! inputs_for.class
      p! inputs_for
      p! input_size_expected
      p! input_sizes_actual
      # p! rnn_bi_di.mini_net_set[sli][tci][channel].weights
      puts

      expect(input_sizes_actual).to eq(input_size_expected)
    end

    context "when NOT passing in any values" do
      # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new }

      context "gathers expected inputs/outputs" do
        context "re :channel_sl_or_combo" do
          it "expected input sizes" do
            # expect(rnn_bi_di.mini_net_set)
            p! rnn_bi_di.input_set_given.class
            p! rnn_bi_di.input_set_given

            expect_input_sizes_channel_sl_or_combo(rnn_bi_di, node_input_sizes_expected)
          end
        end

        it "re :channel_forward" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_forward(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_backwardo" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_backward(rnn_bi_di, node_input_sizes_expected)
        end
      end
    end

    context "when passing in hidden_size_given of 10" do
      let(hidden_size_given) { 10 }

      context "gathers expected inputs/outputs" do
        it "re :channel_sl_or_combo" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_sl_or_combo(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_forward" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_forward(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_backwardo" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_backward(rnn_bi_di, node_input_sizes_expected)
        end
      end
    end

    context "when passing in hidden_layer_qty of 2, hidden_size_given of 10" do
      let(hidden_layer_qty) { 2 }
      let(hidden_size_given) { 10 }
      let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      let(node_input_sizes_expected) {
        [
          [
            # sli: 0
            {
              # sli: 0, tci: 0
              channel_forward: {
                current_self_mem:            0,
                sl_previous_input_or_combo:  0,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             0,
                sl_previous_input_or_combo:   0,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.input_size,
                current_forward:            0,
                current_backward:           0,
              },
            },
            {
              # sli: 0, tci: 1
              channel_forward: {
                current_self_mem:            0,
                sl_previous_input_or_combo:  0,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             0,
                sl_previous_input_or_combo:   0,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.input_size,
                current_forward:            0,
                current_backward:           0,
              },
            },
          ],
          [
            # sli: 1
            {
              # sli: 1, tci: 0
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     rnn_bi_di.hidden_size,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
            {
              # sli: 1, tci: 1
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: rnn_bi_di.hidden_size,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
          ],
          [
            # sli: 2
            {
              # sli: 2, tci: 0
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: rnn_bi_di.hidden_size,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: rnn_bi_di.hidden_size,
                tc_next_channel_backward:     rnn_bi_di.hidden_size,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.output_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
            {
              # sli: 2, tci: 1
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: rnn_bi_di.hidden_size,
                tc_previous_channel_forward: rnn_bi_di.hidden_size,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: rnn_bi_di.hidden_size,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.output_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
          ],
        ]
      }
      let(node_input_sizes_expected) {
        [
          [
            # sli: 0
            {
              # sli: 0, tci: 0
              channel_forward: {
                current_self_mem:            0,
                sl_previous_input_or_combo:  0,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             0,
                sl_previous_input_or_combo:   0,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.input_size,
                current_forward:            0,
                current_backward:           0,
              },
            },
            {
              # sli: 0, tci: 1
              channel_forward: {
                current_self_mem:            0,
                sl_previous_input_or_combo:  0,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             0,
                sl_previous_input_or_combo:   0,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.input_size,
                current_forward:            0,
                current_backward:           0,
              },
            },
          ],
          [
            # sli: 1
            {
              # sli: 1, tci: 0
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     rnn_bi_di.hidden_size,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
            {
              # sli: 1, tci: 1
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: rnn_bi_di.hidden_size,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
          ],
          [
            # sli: 2
            {
              # sli: 2, tci: 0
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: rnn_bi_di.hidden_size,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: rnn_bi_di.hidden_size,
                tc_next_channel_backward:     rnn_bi_di.hidden_size,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.output_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
            {
              # sli: 2, tci: 1
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: rnn_bi_di.hidden_size,
                tc_previous_channel_forward: rnn_bi_di.hidden_size,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: rnn_bi_di.hidden_size,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.output_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
          ],
        ]
      }

      context "gathers expected inputs/outputs" do
        it "re :channel_sl_or_combo" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_sl_or_combo(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_forward" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_forward(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_backwardo" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_backward(rnn_bi_di, node_input_sizes_expected)
        end
      end
    end

    context "when passing in time_col_qty of 3, hidden_layer_qty of 2, hidden_size_given of 10" do
      let(time_col_qty) { 3 }
      let(hidden_layer_qty) { 2 }
      let(hidden_size_given) { 10 }
      let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(time_col_qty: time_col_qty, hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      let(node_input_sizes_expected) {
        [
          [
            # sli: 0
            {
              # sli: 0, tci: 0
              channel_forward: {
                current_self_mem:            0,
                sl_previous_input_or_combo:  0,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             0,
                sl_previous_input_or_combo:   0,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.input_size,
                current_forward:            0,
                current_backward:           0,
              },
            },
            {
              # sli: 0, tci: 1
              channel_forward: {
                current_self_mem:            0,
                sl_previous_input_or_combo:  0,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             0,
                sl_previous_input_or_combo:   0,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.input_size,
                current_forward:            0,
                current_backward:           0,
              },
            },
            {
              # sli: 0, tci: 2
              channel_forward: {
                current_self_mem:            0,
                sl_previous_input_or_combo:  0,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             0,
                sl_previous_input_or_combo:   0,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.input_size,
                current_forward:            0,
                current_backward:           0,
              },
            },
          ],
          [
            # sli: 1
            {
              # sli: 1, tci: 0
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     rnn_bi_di.hidden_size,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
            {
              # sli: 1, tci: 1
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: rnn_bi_di.hidden_size,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     rnn_bi_di.hidden_size,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
            {
              # sli: 1, tci: 2
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: 0,
                tc_previous_channel_forward: rnn_bi_di.hidden_size,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: 0,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.hidden_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
          ],
          [
            # sli: 2
            {
              # sli: 2, tci: 0
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: rnn_bi_di.hidden_size,
                tc_previous_channel_forward: 0,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: rnn_bi_di.hidden_size,
                tc_next_channel_backward:     rnn_bi_di.hidden_size,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.output_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
            {
              # sli: 2, tci: 1
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: rnn_bi_di.hidden_size,
                tc_previous_channel_forward: rnn_bi_di.hidden_size,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: rnn_bi_di.hidden_size,
                tc_next_channel_backward:     rnn_bi_di.hidden_size,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.output_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
            {
              # sli: 2, tci: 2
              channel_forward: {
                current_self_mem:            rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
                sl_previous_channel_forward: rnn_bi_di.hidden_size,
                tc_previous_channel_forward: rnn_bi_di.hidden_size,
              },
              channel_backward: {
                current_self_mem:             rnn_bi_di.hidden_size,
                sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
                sl_previous_channel_backward: rnn_bi_di.hidden_size,
                tc_next_channel_backward:     0,
              },
              channel_sl_or_combo: {
                current_self_mem:           rnn_bi_di.output_size,
                sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                current_forward:            rnn_bi_di.hidden_size,
                current_backward:           rnn_bi_di.hidden_size,
              },
            },
          ],
        ]
      }
      let(input_set_given_example) { [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]] }

      context "gathers expected inputs/outputs" do
        it "re :channel_sl_or_combo" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_sl_or_combo(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_forward" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_forward(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_backwardo" do
          # expect(rnn_bi_di.mini_net_set)
          p! rnn_bi_di.input_set_given.class
          p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_backward(rnn_bi_di, node_input_sizes_expected)
        end
      end
    end
  end
end

# ####
# context "when NOT passing in any values" do
#   # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new }

# end

# context "when passing in hidden_size_given of 10" do
#   let(hidden_size_given) { 10 }

# end

# context "when passing in hidden_layer_qty of 2, hidden_size_given of 10" do
#   let(hidden_layer_qty) { 2 }
#   let(hidden_size_given) { 10 }
#   let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

#   let(node_input_sizes_expected) {
#     [
#       [
#         # sli: 0
#         {
#           # sli: 0, tci: 0
#           channel_forward: {
#             current_self_mem:            0,
#             sl_previous_input_or_combo:  0,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             0,
#             sl_previous_input_or_combo:   0,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.input_size,
#             current_forward:            0,
#             current_backward:           0,
#           },
#         },
#         {
#           # sli: 0, tci: 1
#           channel_forward: {
#             current_self_mem:            0,
#             sl_previous_input_or_combo:  0,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             0,
#             sl_previous_input_or_combo:   0,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.input_size,
#             current_forward:            0,
#             current_backward:           0,
#           },
#         },
#       ],
#       [
#         # sli: 1
#         {
#           # sli: 1, tci: 0
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     rnn_bi_di.hidden_size,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#         {
#           # sli: 1, tci: 1
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: rnn_bi_di.hidden_size,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#       ],
#       [
#         # sli: 2
#         {
#           # sli: 2, tci: 0
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: rnn_bi_di.hidden_size,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: rnn_bi_di.hidden_size,
#             tc_next_channel_backward:     rnn_bi_di.hidden_size,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.output_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#         {
#           # sli: 2, tci: 1
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: rnn_bi_di.hidden_size,
#             tc_previous_channel_forward: rnn_bi_di.hidden_size,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: rnn_bi_di.hidden_size,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.output_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#       ],
#     ]
#   }
# end

# context "when passing in time_col_qty of 3, hidden_layer_qty of 2, hidden_size_given of 10" do
#   let(time_col_qty) { 3 }
#   let(hidden_layer_qty) { 2 }
#   let(hidden_size_given) { 10 }
#   let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(time_col_qty: time_col_qty, hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

#   let(node_input_sizes_expected) {
#     [
#       [
#         # sli: 0
#         {
#           # sli: 0, tci: 0
#           channel_forward: {
#             current_self_mem:            0,
#             sl_previous_input_or_combo:  0,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             0,
#             sl_previous_input_or_combo:   0,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.input_size,
#             current_forward:            0,
#             current_backward:           0,
#           },
#         },
#         {
#           # sli: 0, tci: 1
#           channel_forward: {
#             current_self_mem:            0,
#             sl_previous_input_or_combo:  0,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             0,
#             sl_previous_input_or_combo:   0,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.input_size,
#             current_forward:            0,
#             current_backward:           0,
#           },
#         },
#         {
#           # sli: 0, tci: 2
#           channel_forward: {
#             current_self_mem:            0,
#             sl_previous_input_or_combo:  0,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             0,
#             sl_previous_input_or_combo:   0,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.input_size,
#             current_forward:            0,
#             current_backward:           0,
#           },
#         },
#       ],
#       [
#         # sli: 1
#         {
#           # sli: 1, tci: 0
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     rnn_bi_di.hidden_size,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#         {
#           # sli: 1, tci: 1
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: rnn_bi_di.hidden_size,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     rnn_bi_di.hidden_size,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#         {
#           # sli: 1, tci: 2
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: 0,
#             tc_previous_channel_forward: rnn_bi_di.hidden_size,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: 0,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#       ],
#       [
#         # sli: 2
#         {
#           # sli: 2, tci: 0
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: rnn_bi_di.hidden_size,
#             tc_previous_channel_forward: 0,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: rnn_bi_di.hidden_size,
#             tc_next_channel_backward:     rnn_bi_di.hidden_size,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.output_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#         {
#           # sli: 2, tci: 1
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: rnn_bi_di.hidden_size,
#             tc_previous_channel_forward: rnn_bi_di.hidden_size,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: rnn_bi_di.hidden_size,
#             tc_next_channel_backward:     rnn_bi_di.hidden_size,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.output_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#         {
#           # sli: 2, tci: 2
#           channel_forward: {
#             current_self_mem:            rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:  rnn_bi_di.hidden_size,
#             sl_previous_channel_forward: rnn_bi_di.hidden_size,
#             tc_previous_channel_forward: rnn_bi_di.hidden_size,
#           },
#           channel_backward: {
#             current_self_mem:             rnn_bi_di.hidden_size,
#             sl_previous_input_or_combo:   rnn_bi_di.hidden_size,
#             sl_previous_channel_backward: rnn_bi_di.hidden_size,
#             tc_next_channel_backward:     0,
#           },
#           channel_sl_or_combo: {
#             current_self_mem:           rnn_bi_di.output_size,
#             sl_previous_input_or_combo: rnn_bi_di.hidden_size,
#             current_forward:            rnn_bi_di.hidden_size,
#             current_backward:           rnn_bi_di.hidden_size,
#           },
#         },
#       ],
#     ]
#   }

# end
