require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::PropsAndInits do
  # NOTE: This also tests Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::PaiDistinct. Maybe those tests should be pulled out?

  def expect_mini_net_set(rnn_bi_di)
    # sli_last = rnn_bi_di.synaptic_layer_indexes.last
    rnn_bi_di.synaptic_layer_indexes.map do |sli|
      rnn_bi_di.time_col_indexes.map do |tci|
        channels = [:channel_forward, :channel_backward, :channel_input_or_combo]
        channels.each do |channel_symbol|
          # puts "v"*80
          expect(channel_symbol).to be_a(Symbol)

          # p! sli
          # p! tci
          # p! channel_symbol
          # if rnn_bi_di.mini_net_set[sli][tci].keys.includes?(channel_symbol)
          #   p! rnn_bi_di.mini_net_set[sli][tci][channel_symbol]
          #   puts "MiniNet at: [#{sli}][#{tci}][#{channel_symbol}]"
          # else
          #   puts "MiniNet NOT at: [#{sli}][#{tci}][#{channel_symbol}]"
          # end

          if sli == 0 && channel_symbol != :channel_input_or_combo
            expect(rnn_bi_di.mini_net_set[sli][tci].keys).not_to contain(channel_symbol)
          else
            expect(rnn_bi_di.mini_net_set[sli][tci].keys).to contain(channel_symbol)

            mn_input_sizes = rnn_bi_di.node_input_sizes[sli][tci][channel_symbol]
            expected_input_size_total = mn_input_sizes.values.sum
            mini_net = rnn_bi_di.mini_net_set[sli][tci][channel_symbol]

            expect(mini_net.height).to eq(expected_input_size_total)
          end
          # puts "^"*80
        end
      end
    end
  end

  describe "#initialize" do
    let(hidden_size_given) { 0 } # aka the default
    let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

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
            channel_input_or_combo: {
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
            channel_input_or_combo: {
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
            channel_input_or_combo: {
              current_self_mem:           rnn_bi_di.output_size,
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
            channel_input_or_combo: {
              current_self_mem:           rnn_bi_di.output_size,
              sl_previous_input_or_combo: rnn_bi_di.hidden_size,
              current_forward:            rnn_bi_di.hidden_size,
              current_backward:           rnn_bi_di.hidden_size,
            },
          },
        ],
      ]
    }

    context "when NOT passing in any values" do
      # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new }

      it "has no errors" do
        expect(rnn_bi_di.errors.empty?).to be_true
        expect(rnn_bi_di.errors.is_a?(Hash(String, String))).to be_true
        expect(rnn_bi_di.errors).to eq(Hash(String, String).new)
      end

      it "is valid" do
        expect(rnn_bi_di.valid?).to be_true
      end

      context "has expected value for property" do
        it "time_col_qty" do
          expect(rnn_bi_di.time_col_qty).to eq(2)
        end

        it "input_size" do
          expect(rnn_bi_di.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_bi_di.output_size).to eq(1)
        end

        it "hidden_layer_qty" do
          expect(rnn_bi_di.hidden_layer_qty).to eq(1)
        end

        it "hidden_size" do
          expect(rnn_bi_di.hidden_size).to eq(3)
        end

        it "synaptic_layer_qty" do
          expect(rnn_bi_di.synaptic_layer_qty).to eq(2)
        end

        it "@synaptic_layer_indexes" do
          expect(rnn_bi_di.synaptic_layer_indexes).to eq([0, 1])
        end

        it "@node_input_sizes" do
          puts "*"*80
          puts "rnn_bi_di.inputs_size: #{rnn_bi_di.input_size}"
          puts
          puts "rnn_bi_di.hidden_size: #{rnn_bi_di.hidden_size}"
          puts
          puts "rnn_bi_di.output_size: #{rnn_bi_di.output_size}"
          puts
          puts "rnn_bi_di.node_input_sizes.pretty_inspect:\n#{rnn_bi_di.node_input_sizes.pretty_inspect}"
          puts
          puts "node_input_sizes_expected.pretty_inspect: #{node_input_sizes_expected.pretty_inspect}"
          puts
          # puts "COMPARE: " + ("v"*80)
          # puts({actual: rnn_bi_di.node_input_sizes, expected: node_input_sizes_expected}.to_json)
          # puts "COMPARE: " + ("^"*80)
          # puts
          puts "-"*80
          expect(rnn_bi_di.node_input_sizes).to eq(node_input_sizes_expected)
        end
      end

      context "mini_net_set" do
        # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

        it "each mini_net is of the expected width and height" do
          expect_mini_net_set(rnn_bi_di)
        end
      end
    end

    context "when passing in hidden_size_given of 10" do
      let(hidden_size_given) { 10 }
      # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

      it "has no errors" do
        expect(rnn_bi_di.errors.empty?).to be_true
        expect(rnn_bi_di.errors.is_a?(Hash(String, String))).to be_true
        expect(rnn_bi_di.errors).to eq(Hash(String, String).new)
      end

      it "is valid" do
        expect(rnn_bi_di.valid?).to be_true
      end

      context "has expected value for property" do
        it "time_col_qty" do
          expect(rnn_bi_di.time_col_qty).to eq(2)
        end

        it "input_size" do
          expect(rnn_bi_di.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_bi_di.output_size).to eq(1)
        end

        it "hidden_layer_qty" do
          expect(rnn_bi_di.hidden_layer_qty).to eq(1)
        end

        it "when hidden_size_given is 0, hidden_size uses defaults" do
          expect(hidden_size_given).to eq(10)
          expect(rnn_bi_di.hidden_size).to eq(hidden_size_given)
          expect(rnn_bi_di.hidden_size).not_to eq(rnn_bi_di.input_size + rnn_bi_di.output_size)
        end

        it "synaptic_layer_qty" do
          expect(rnn_bi_di.synaptic_layer_qty).to eq(2)
        end

        it "@synaptic_layer_indexes" do
          expect(rnn_bi_di.synaptic_layer_indexes).to eq([0, 1])
        end

        it "@node_input_sizes" do
          puts "*"*80
          puts "rnn_bi_di.inputs_size: #{rnn_bi_di.input_size}"
          puts
          puts "rnn_bi_di.hidden_size: #{rnn_bi_di.hidden_size}"
          puts
          puts "rnn_bi_di.output_size: #{rnn_bi_di.output_size}"
          puts
          puts "rnn_bi_di.node_input_sizes.pretty_inspect:\n#{rnn_bi_di.node_input_sizes.pretty_inspect}"
          puts
          puts "node_input_sizes_expected.pretty_inspect: #{node_input_sizes_expected.pretty_inspect}"
          puts
          puts "-"*80
          expect(rnn_bi_di.node_input_sizes).to eq(node_input_sizes_expected)
        end
      end

      context "mini_net_set" do
        # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

        it "each mini_net is of the expected width and height" do
          expect_mini_net_set(rnn_bi_di)
        end
      end
    end

    context "when passing in hidden_layer_qty of 2, hidden_size_given of 10" do
      let(hidden_layer_qty) { 2 }
      let(hidden_size_given) { 10 }
      let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      it "has no errors" do
        expect(rnn_bi_di.errors.empty?).to be_true
        expect(rnn_bi_di.errors.is_a?(Hash(String, String))).to be_true
        expect(rnn_bi_di.errors).to eq(Hash(String, String).new)
      end

      it "is valid" do
        expect(rnn_bi_di.valid?).to be_true
      end

      context "has expected value for property" do
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
                  current_self_mem:           rnn_bi_di.output_size,
                  sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                  current_forward:            rnn_bi_di.hidden_size,
                  current_backward:           rnn_bi_di.hidden_size,
                },
              },
            ],
          ]
        }

        it "time_col_qty" do
          expect(rnn_bi_di.time_col_qty).to eq(2)
        end

        it "input_size" do
          expect(rnn_bi_di.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_bi_di.output_size).to eq(1)
        end

        it "hidden_layer_qty" do
          expect(rnn_bi_di.hidden_layer_qty).to eq(hidden_layer_qty)
        end

        it "hidden_size" do
          expect(rnn_bi_di.hidden_size).to eq(hidden_size_given)
        end

        it "synaptic_layer_qty" do
          expect(rnn_bi_di.synaptic_layer_qty).to eq(hidden_layer_qty + 1)
        end

        it "@synaptic_layer_indexes" do
          expect(rnn_bi_di.synaptic_layer_indexes).to eq((0..hidden_layer_qty).to_a)
        end

        it "@node_input_sizes" do
          puts "*"*80
          puts "rnn_bi_di.inputs_size: #{rnn_bi_di.input_size}"
          puts
          puts "rnn_bi_di.hidden_size: #{rnn_bi_di.hidden_size}"
          puts
          puts "rnn_bi_di.output_size: #{rnn_bi_di.output_size}"
          puts
          puts "rnn_bi_di.node_input_sizes.pretty_inspect:\n#{rnn_bi_di.node_input_sizes.pretty_inspect}"
          puts
          puts "node_input_sizes_expected.pretty_inspect: #{node_input_sizes_expected.pretty_inspect}"
          puts
          puts "-"*80
          expect(rnn_bi_di.node_input_sizes).to eq(node_input_sizes_expected)
        end
      end

      context "mini_net_set" do
        # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

        it "each mini_net is of the expected width and height" do
          expect_mini_net_set(rnn_bi_di)
        end
      end
    end

    context "when passing in time_col_qty of 3, hidden_layer_qty of 2, hidden_size_given of 10" do
      let(time_col_qty) { 3 }
      let(hidden_layer_qty) { 2 }
      let(hidden_size_given) { 10 }
      let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(time_col_qty: time_col_qty, hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      it "has no errors" do
        expect(rnn_bi_di.errors.empty?).to be_true
        expect(rnn_bi_di.errors.is_a?(Hash(String, String))).to be_true
        expect(rnn_bi_di.errors).to eq(Hash(String, String).new)
      end

      it "is valid" do
        expect(rnn_bi_di.valid?).to be_true
      end

      context "has expected value for property" do
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
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
                channel_input_or_combo: {
                  current_self_mem:           rnn_bi_di.output_size,
                  sl_previous_input_or_combo: rnn_bi_di.hidden_size,
                  current_forward:            rnn_bi_di.hidden_size,
                  current_backward:           rnn_bi_di.hidden_size,
                },
              },
            ],
          ]
        }

        it "time_col_qty" do
          expect(rnn_bi_di.time_col_qty).to eq(time_col_qty)
        end

        it "input_size" do
          expect(rnn_bi_di.input_size).to eq(2)
        end

        it "output_size" do
          expect(rnn_bi_di.output_size).to eq(1)
        end

        it "hidden_layer_qty" do
          expect(rnn_bi_di.hidden_layer_qty).to eq(hidden_layer_qty)
        end

        it "hidden_size" do
          expect(rnn_bi_di.hidden_size).to eq(hidden_size_given)
        end

        it "synaptic_layer_qty" do
          expect(rnn_bi_di.synaptic_layer_qty).to eq(hidden_layer_qty + 1)
        end

        it "@synaptic_layer_indexes" do
          expect(rnn_bi_di.synaptic_layer_indexes).to eq((0..hidden_layer_qty).to_a)
        end

        it "@node_input_sizes" do
          puts "*"*80
          puts "rnn_bi_di.inputs_size: #{rnn_bi_di.input_size}"
          puts
          puts "rnn_bi_di.hidden_size: #{rnn_bi_di.hidden_size}"
          puts
          puts "rnn_bi_di.output_size: #{rnn_bi_di.output_size}"
          puts
          puts "rnn_bi_di.node_input_sizes.pretty_inspect:\n#{rnn_bi_di.node_input_sizes.pretty_inspect}"
          puts
          puts "node_input_sizes_expected.pretty_inspect: #{node_input_sizes_expected.pretty_inspect}"
          puts
          puts "-"*80
          expect(rnn_bi_di.node_input_sizes).to eq(node_input_sizes_expected)
        end
      end

      context "mini_net_set" do
        # let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

        it "each mini_net is of the expected width and height" do
          expect_mini_net_set(rnn_bi_di)
        end
      end
    end

    # it "just some debugging" do # TODO: REMOVE before merging!
    #   # puts rnn_bi_di.to_pretty_json

    #   # rnn_bi_di.nodal_layer_indexes.map do |li|
    #   rnn_bi_di.synaptic_layer_indexes.map do |li|
    #     rnn_bi_di.time_col_indexes.map do |ti|
    #       # debug_info = {"li": li, "ti": ti, "rnn_bi_di.node_input_sizes[li][ti]": rnn_bi_di.node_input_sizes[li][ti]}
    #       debug_info = {"li": li, "ti": ti, "node_input_sizes": rnn_bi_di.node_input_sizes[li][ti]}
    #       # puts debug_info.to_json
    #     end
    #   end
    # end

    # context "mini_net_set" do
    #   let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new }

    #   it "each mini_net is of the expected width and height" do
    #     rnn_bi_di.synaptic_layer_indexes.map do |li|
    #       rnn_bi_di.time_col_indexes.map do |ti|
    #         mini_net = rnn_bi_di.mini_net_set[li][ti]

    #         expected_input_size_total = rnn_bi_di.node_input_sizes[li][ti].values.sum
    #         expect(mini_net.height).to eq(expected_input_size_total)

    #         expected_output_size = rnn_bi_di.node_output_sizes[li]
    #         expect(mini_net.width).to eq(expected_output_size)
    #       end
    #     end
    #   end
    # end
  end

  # # sli: 0
  # {
  #   # sli: 0, tci: 0
  #   channel_forward: {
  #     current_self_mem:            0,
  #     sl_previous_input_or_combo:  0,
  #     sl_previous_channel_forward: 0,
  #     tc_previous_channel_forward: 0,
  #   },
  #   channel_backward: {
  #     current_self_mem:             0,
  #     sl_previous_input_or_combo:   0,
  #     sl_previous_channel_backward: 0,
  #     tc_next_channel_backward:     0,
  #   },
  #   channel_input_or_combo: {
  #     current_self_mem:           rnn_bi_di.hidden_size,
  #     sl_previous_input_or_combo: rnn_bi_di.input_size,
  #     current_forward:            0,
  #     current_backward:           0,
  #   },
  # },

end
