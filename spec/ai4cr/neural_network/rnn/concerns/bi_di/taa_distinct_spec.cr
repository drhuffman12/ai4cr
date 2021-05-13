require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::Concerns::BiDi::TaaDistinct do
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

  let(output_set_expected) { [[0.9], [0.1]] }

  let(outputs_guessed_expected) { [[0.05910000000000001], [0.041186]] }

  describe "#outputs_for" do
    let(outputs_for) { rnn_bi_di.outputs_for(sli, tci, channel) }

    before_each do
      rnn_bi_di.input_set_given = input_set_given_example
      rnn_bi_di.eval(input_set_given_example)
      rnn_bi_di.output_set_expected = output_set_expected
    end

    context "when NOT passing in any values to RnnBiDi" do
      context "generated values for" do
        it "synaptic_layer_indexes" do
          expect(rnn_bi_di.synaptic_layer_indexes).to eq([0, 1])
        end

        it "time_col_indexes" do
          expect(rnn_bi_di.time_col_indexes).to eq([0, 1])
        end
      end

      context "for sli '0' tci '0' channel ':channel_sl_or_combo'" do
        let(sli) { 0 }
        let(tci) { 0 }
        let(channel) {
          :channel_sl_or_combo
        }

        context "returns" do
          it "expected class" do
            puts
            puts "v"*80
            p! outputs_for
            puts "^"*80
            puts

            class_expected = NamedTuple(outs_deltas: Hash(Symbol, Array(Float64)), outs_expected: Array(Float64))
            expect(outputs_for.class).to eq(class_expected)
          end

          context "expected nested sizes for key" do
            it ":outs_deltas" do
              expect(outputs_for[:outs_deltas].keys).to eq([:current_self_mem, :sl_next_channel_forward, :sl_next_channel_backward, :sl_next_channel_combo])
            end
            it ":outs_expected" do
              expect(outputs_for[:outs_expected].size).to eq(0)
            end
          end
        end
      end

      context "for sli '0' tci '1' channel ':channel_sl_or_combo'" do
        let(sli) { 0 }
        let(tci) { 1 }
        let(channel) {
          :channel_sl_or_combo
        }

        context "returns" do
          it "expected class" do
            puts
            puts "v"*80
            p! outputs_for
            puts "^"*80
            puts

            class_expected = NamedTuple(outs_deltas: Hash(Symbol, Array(Float64)), outs_expected: Array(Float64))
            expect(outputs_for.class).to eq(class_expected)
          end

          context "expected nested sizes for key" do
            it ":outs_deltas" do
              expect(outputs_for[:outs_deltas].keys).to eq([:current_self_mem, :sl_next_channel_forward, :sl_next_channel_backward, :sl_next_channel_combo])
            end
            it ":outs_expected" do
              expect(outputs_for[:outs_expected].size).to eq(0)
            end
          end
        end
      end

      context "for sli '1' tci '0' channel ':channel_sl_or_combo'" do
        let(sli) { 1 }
        let(tci) { 0 }
        let(channel) {
          :channel_sl_or_combo
        }

        context "returns" do
          it "expected class" do
            puts
            puts "v"*80
            p! outputs_for
            p! outputs_for.class
            puts "^"*80
            puts

            class_expected = NamedTuple(outs_deltas: Hash(Symbol, Array(Float64)), outs_expected: Array(Float64))
            expect(outputs_for.class).to eq(class_expected)
          end

          context "expected nested sizes for key" do
            it ":outs_deltas" do
              expect(outputs_for[:outs_deltas].keys).to eq([:current_self_mem])
            end
            it ":outs_expected" do
              expect(outputs_for[:outs_expected].size).to eq(1)
            end
          end
        end
      end

      context "for sli '1' tci '1' channel ':channel_sl_or_combo'" do
        let(sli) { 1 }
        let(tci) { 1 }
        let(channel) {
          :channel_sl_or_combo
        }

        context "returns" do
          it "expected class" do
            puts
            puts "v"*80
            p! outputs_for
            p! outputs_for.class
            puts "^"*80
            puts

            class_expected = NamedTuple(outs_deltas: Hash(Symbol, Array(Float64)), outs_expected: Array(Float64))
            expect(outputs_for.class).to eq(class_expected)
          end

          context "expected nested sizes for key" do
            it ":outs_deltas" do
              expect(outputs_for[:outs_deltas].keys).to eq([:current_self_mem])
            end
            it ":outs_expected" do
              expect(outputs_for[:outs_expected].size).to eq(1)
            end
          end
        end
      end

      context "returns" do
        let(class_expected) {
          # TODO: Why the '| Symbol' in the 'Hash'?
          Array(Array(Hash(NamedTuple(outs_deltas: Hash(Symbol, Array(Float64)), outs_expected: Array(Float64)) | Symbol, NamedTuple(outs_deltas: Hash(Symbol, Array(Float64)), outs_expected: Array(Float64)) | Symbol)))
        }

        let(outputs_for_all_expected) {
          [
            [
              {
                :channel_sl_or_combo => {
                  outs_deltas: {
                    :current_self_mem         => [0.0, 0.0, 0.0],
                    :sl_next_channel_forward  => [0.0, 0.0, 0.0],
                    :sl_next_channel_backward => [0.0, 0.0, 0.0],
                    :sl_next_channel_combo    => [0.0, 0.0, 0.0],
                  },
                  outs_expected: [] of Float64,
                },
              },
              {
                :channel_sl_or_combo => {
                  outs_deltas: {
                    :current_self_mem         => [0.0, 0.0, 0.0],
                    :sl_next_channel_forward  => [0.0, 0.0, 0.0],
                    :sl_next_channel_backward => [0.0, 0.0, 0.0],
                    :sl_next_channel_combo    => [0.0, 0.0, 0.0],
                  },
                  outs_expected: [] of Float64,
                },
              },
            ],
            [
              {
                :channel_forward => {
                  outs_deltas: {
                    :current_self_mem => [0.0, 0.0, 0.0],
                  },
                  outs_expected: [] of Float64,
                },
                :channel_backward => {
                  outs_deltas: {
                    :current_self_mem => [0.0, 0.0, 0.0],
                  },
                  outs_expected: [] of Float64,
                },
                :channel_sl_or_combo => {
                  outs_deltas: {
                    :current_self_mem => [0.0],
                  },
                  outs_expected: [0.9],
                },
              },
              {
                :channel_forward => {
                  outs_deltas: {
                    :current_self_mem => [0.0, 0.0, 0.0],
                  },
                  outs_expected: [] of Float64,
                },
                :channel_backward => {
                  outs_deltas: {
                    :current_self_mem => [0.0, 0.0, 0.0],
                  },
                  outs_expected: [] of Float64,
                },
                :channel_sl_or_combo => {
                  outs_deltas: {
                    :current_self_mem => [0.0],
                  },
                  outs_expected: [0.1],
                },
              },
            ],
          ]
        }
        it "expected class" do
          # This doesn't collect the inputs in the right order, so the values won't be correct.
          #   BUT, the configuration should be correct .. so we can test it:
          outputs_for_all = rnn_bi_di.map_only_indexes do |sli, tci, channel|
            puts
            puts "v"*80
            p! [sli, tci, channel]
            puts "^"*80
            puts

            rnn_bi_di.outputs_for(sli, tci, channel)
          end

          puts
          puts "v"*80
          p! outputs_for_all
          puts "^"*80
          puts

          expect(outputs_for_all.class).to eq(class_expected)
        end
      end
    end

    context "when passing in hidden_size_given of 10" do
      let(hidden_size_given) { 10 }

      pending "todo" do
        expect(1).to eq(0)
      end
    end

    context "when passing in hidden_layer_qty of 2, hidden_size_given of 10" do
      let(hidden_layer_qty) { 2 }
      let(hidden_size_given) { 10 }
      let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      pending "todo" do
        expect(1).to eq(0)
      end
    end

    context "when passing in time_col_qty of 3, hidden_layer_qty of 2, hidden_size_given of 10" do
      let(time_col_qty) { 3 }
      let(hidden_layer_qty) { 2 }
      let(hidden_size_given) { 10 }
      let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(time_col_qty: time_col_qty, hidden_layer_qty: hidden_layer_qty, hidden_size_given: hidden_size_given) }

      pending "todo" do
        expect(1).to eq(0)
      end
    end
  end
end
