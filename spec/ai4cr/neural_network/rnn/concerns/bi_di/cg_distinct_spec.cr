require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::CgDistinct do
  let(hidden_size_given) { 0 } # aka the default (which leads to an actual hidden size of '1')
  let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

  let(input_set_given_example) { [[10.1, 10.2], [10.3, 10.4]] }

  let(weights_example) {
    [
      [
        {
          :channel_sl_or_combo => [
            [10.01, 0.02, -0.03],
            [0.04, -10.05, 0.06],
            [0.07, 0.08, 10.09],
            [-10.10, 0.11, -0.12],
            [-0.13, 10.14, -10.15],
            [10.16, -0.17, -0.18],
          ],
        },
        {
          :channel_sl_or_combo => [
            [-0.01, -10.02, 0.03],
            [0.04, -0.05, 10.06],
            [10.07, 0.08, 0.09],
            [0.10, 10.11, -0.12],
            [0.13, -0.14, -10.15],
            [-10.16, 0.17, 0.18],
          ],
        },
      ],
      [
        {
          :channel_forward => [
            [0.01, -10.02, 0.03],
            [-0.04, -0.05, -10.06],
            [-10.07, -0.08, 0.09],
            [0.10, -10.11, -0.12],
            [0.13, -0.14, 10.15],
            [10.16, 0.17, 0.18],
          ],
          :channel_backward => [
            [0.01, 10.02, 0.03],
            [0.04, -0.05, -10.06],
            [-10.07, 0.08, -0.09],
            [-0.10, -10.11, -0.12],
            [-0.13, 0.14, -10.15],
            [10.16, -0.17, -0.18],
            [0.19, -10.20, -0.21],
            [0.22, -0.23, -10.24],
            [10.25, -0.26, -0.27],
          ],
          :channel_sl_or_combo => [
            [0.01],
            [0.02],
            [-0.03],
            [-0.04],
            [0.05],
            [0.06],
            [0.07],
            [0.08],
            [0.09],
            [0.10],
          ],
        },
        {
          :channel_forward => [
            [-0.01, 0.02, 0.03],
            [-0.04, 0.05, -0.06],
            [0.07, 0.08, -0.09],
            [-0.10, 0.11, -0.12],
            [-0.13, -0.14, -0.15],
            [0.16, 0.17, 0.18],
            [-0.19, -0.20, 0.21],
            [-0.22, 0.23, -0.24],
            [0.25, 0.26, 0.27],
          ],
          :channel_backward => [
            [0.01, -0.02, 0.03],
            [-0.04, 0.05, -0.06],
            [-0.07, 0.08, 0.09],
            [-0.10, 0.11, -0.12],
            [-0.13, -0.14, -0.15],
            [-0.16, 0.17, 0.18],
          ],
          :channel_sl_or_combo => [
            [0.01],
            [-0.02],
            [0.03],
            [-0.04],
            [-0.05],
            [0.06],
            [0.07],
            [-0.08],
            [0.09],
            [-0.10],
          ],
        },
      ],
    ]
  }

  let(outputs_guessed_expected) { [[0.05910000000000001], [0.041186]] }

  let(all_outputs_expected_before) {
    [
      [
        {
          :channel_sl_or_combo => [0.0, 0.0, 0.0],
        },
        {
          :channel_sl_or_combo => [0.0, 0.0, 0.0],
        },
      ],
      [
        {
          :channel_forward     => [0.0, 0.0, 0.0],
          :channel_backward    => [0.0, 0.0, 0.0],
          :channel_sl_or_combo => [0.0],
        },
        {
          :channel_forward     => [0.0, 0.0, 0.0],
          :channel_backward    => [0.0, 0.0, 0.0],
          :channel_sl_or_combo => [0.0],
        },
      ],
    ]
  }
  let(all_outputs_expected_after) {
    [
      [
        {
          :channel_sl_or_combo => [0.0, 1.0, 0.0],
        },
        {
          :channel_sl_or_combo => [0.0, 1.0, 0.0],
        },
      ],
      [
        {
          :channel_forward     => [0.13, 0.0, 1.0],
          :channel_backward    => [0.0, 0.14, 0.0],
          :channel_sl_or_combo => [0.05910000000000001],
        },
        {
          :channel_forward     => [0.0953, 0.094, 0.14730000000000004],
          :channel_backward    => [0.0, 0.0, 0.0],
          :channel_sl_or_combo => [0.041186],
        },
      ],
    ]
  }

  describe "#eval" do
    context "when using a RnnBiDi initialized without passing in any values" do
      it "updates outputs of each mini_net as expected" do
        expect(rnn_bi_di.all_mini_net_outputs).to eq(all_outputs_expected_before)
        rnn_bi_di.weights = weights_example
        rnn_bi_di.eval(input_set_given_example)
        expect(rnn_bi_di.all_mini_net_outputs).to eq(all_outputs_expected_after)
        expect(all_outputs_expected_before).not_to eq(all_outputs_expected_after)
      end

      context "returns" do
        it "expected weights" do
          # set/mock weights for rnn_bi_di
          rnn_bi_di.weights = weights_example

          # puts
          # puts "BEFORE:"
          # puts "rnn_bi_di.weights.pretty_inspect:"
          # puts rnn_bi_di.weights.pretty_inspect
          # puts

          # eval
          rnn_bi_di.eval(input_set_given_example)

          # puts
          # puts "AFTER:"
          # puts "rnn_bi_di.weights.pretty_inspect:"
          # puts rnn_bi_di.weights.pretty_inspect
          # puts

          # compare
          expect(rnn_bi_di.outputs_guessed).to eq(outputs_guessed_expected)
        end
      end
    end
  end

  describe "#inputs_for" do
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

      # puts
      # p! sli
      # p! tci
      # p! channel
      # p! node_input_sizes_expected[sli][tci][channel]
      # p! inputs_for.class
      # p! inputs_for
      # p! input_size_expected
      # p! input_sizes_actual
      # p! rnn_bi_di.mini_net_set[sli][tci][channel].weights
      # puts

      expect(input_sizes_actual).to eq(input_size_expected)
    end

    context "when NOT passing in any values" do
      # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new }

      context "gathers expected inputs/outputs" do
        context "re :channel_sl_or_combo" do
          it "expected input sizes" do
            # p! rnn_bi_di.input_set_given.class
            # p! rnn_bi_di.input_set_given

            expect_input_sizes_channel_sl_or_combo(rnn_bi_di, node_input_sizes_expected)
          end
        end

        it "re :channel_forward" do
          # p! rnn_bi_di.input_set_given.class
          # p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_forward(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_backwardo" do
          # p! rnn_bi_di.input_set_given.class
          # p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_backward(rnn_bi_di, node_input_sizes_expected)
        end
      end
    end

    context "when passing in hidden_size_given of 10" do
      let(hidden_size_given) { 10 }

      context "gathers expected inputs/outputs" do
        it "re :channel_sl_or_combo" do
          # p! rnn_bi_di.input_set_given.class
          # p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_sl_or_combo(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_forward" do
          # p! rnn_bi_di.input_set_given.class
          # p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_forward(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_backwardo" do
          # p! rnn_bi_di.input_set_given.class
          # p! rnn_bi_di.input_set_given

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
          # p! rnn_bi_di.input_set_given.class
          # p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_sl_or_combo(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_forward" do
          # p! rnn_bi_di.input_set_given.class
          # p! rnn_bi_di.input_set_given

          expect_input_sizes_channel_forward(rnn_bi_di, node_input_sizes_expected)
        end

        it "re :channel_backwardo" do
          # p! rnn_bi_di.input_set_given.class
          # p! rnn_bi_di.input_set_given

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

  describe "#weights" do
    context "returns expected " do
      pending "class" do
        # actual: Array(Array(Hash(Array(Array(Float64)) | Symbol, Array(Array(Float64)) | Symbol)))
        # expected: Array(Array(Hash(Symbol, Array(Array(Float64)))))

        # puts
        # puts "rnn_bi_di.weights.pretty_inspect:"
        # puts rnn_bi_di.weights.pretty_inspect
        # puts
        # puts "rnn_bi_di.weights.class: #{rnn_bi_di.weights.class}"
        # puts
        expect(rnn_bi_di.weights.class).to eq(Array(Array(Hash(Symbol, Array(Array(Float64))))))
        expect(rnn_bi_di.weights.class).to eq(Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::Weights)
      end

      it "values" do
        # puts
        # puts "BEFORE:"
        # puts "rnn_bi_di.weights.pretty_inspect:"
        # puts rnn_bi_di.weights.pretty_inspect
        # puts
        rnn_bi_di.weights = weights_example
        # puts
        # puts "AFTER:"
        # puts "rnn_bi_di.weights.pretty_inspect:"
        # puts rnn_bi_di.weights.pretty_inspect
        # puts
        # puts "rnn_bi_di.weights.class: #{rnn_bi_di.weights.class}"
        # puts
        expect(rnn_bi_di.weights).to eq(weights_example)
      end
    end
  end

  describe "#weights=(w)" do
    context "returns expected " do
      pending "class" do
        # actual: Array(Array(Hash(Array(Array(Float64)) | Symbol, Array(Array(Float64)) | Symbol)))
        # expected: Array(Array(Hash(Symbol, Array(Array(Float64)))))

        weights_before = rnn_bi_di.weights.clone
        rnn_bi_di.weights = weights_example
        weights_after = rnn_bi_di.weights.clone

        expect(weights_example.class).to eq(Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::Weights)
        expect(weights_before.class).to eq(Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::Weights)
        expect(weights_after.class).to eq(Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::Weights)
      end

      it "value" do
        weights_before = rnn_bi_di.weights.clone
        puts
        puts "BEFORE:"
        puts "rnn_bi_di.weights.pretty_inspect:"
        puts rnn_bi_di.weights.pretty_inspect
        puts
        rnn_bi_di.weights = weights_example
        puts
        puts "AFTER:"
        puts "rnn_bi_di.weights.pretty_inspect:"
        puts rnn_bi_di.weights.pretty_inspect
        puts

        weights_after = rnn_bi_di.weights.clone

        expect(weights_before).not_to eq(weights_example)
        expect(weights_after).to eq(weights_example)
      end
    end
  end

  describe "#all_mini_net_outputs" do
    context "returns" do
      it "expected values" do
        expect(rnn_bi_di.all_mini_net_outputs).to eq(all_outputs_expected_before)
      end
    end
  end
end
