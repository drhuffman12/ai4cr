require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Rnn::RnnBiDiConcerns::PropsAndInits do
  describe "#initialize" do
    let(hidden_size_given) { 0 } # aka the default
    let(rnn_bi_di) { Ai4cr::NeuralNetwork::Rnn::RnnBiDi.new(hidden_size_given: hidden_size_given) }

    let(node_input_sizes_expected) {
      [
        # sli: 0
        [
          # tci: 0
          {
            # channels
            channel_forward: {
              # aka 'disabled'
              current_self_mem:                            0,
              previous_synaptic_layer_channel_sl_or_combo: 0,
              previous_synaptic_layer_channel_forward:     0,
              previous_time_column:                        0,
            },
            channel_backward: {
              # aka 'disabled'
              current_self_mem:                            0,
              previous_synaptic_layer_channel_sl_or_combo: 0,
              previous_synaptic_layer_channel_backward:    0,
              next_time_column:                            0,
            },
            channel_sl_or_combo: {
              current_self_mem:                            rnn_bi_di.hidden_size,
              previous_synaptic_layer_channel_sl_or_combo: rnn_bi_di.input_size,
              current_forward:                             0,
              current_backward:                            0,
            },
          },
          # tci: 1
          {
            # channels
            channel_forward: {
              # aka 'disabled'
              current_self_mem:                            0,
              previous_synaptic_layer_channel_sl_or_combo: 0,
              previous_synaptic_layer_channel_forward:     0,
              previous_time_column:                        0,
            },
            channel_backward: {
              # aka 'disabled'
              current_self_mem:                            0,
              previous_synaptic_layer_channel_sl_or_combo: 0,
              previous_synaptic_layer_channel_backward:    0,
              next_time_column:                            0,
            },
            channel_sl_or_combo: {
              current_self_mem:                            rnn_bi_di.hidden_size,
              previous_synaptic_layer_channel_sl_or_combo: rnn_bi_di.input_size,
              current_forward:                             0,
              current_backward:                            0,
            },
          },
        ],
        # sli: 1
        [
          # tci: 0
          {
            # channels
            channel_forward: {
              current_self_mem:                            rnn_bi_di.output_size, # varies per sli
              previous_synaptic_layer_channel_sl_or_combo: rnn_bi_di.hidden_size, # varies per sli
              previous_synaptic_layer_channel_forward:     0,                     # varies per sli and tci
              previous_time_column:                        0,                     # varies per sli and tci
            },
            channel_backward: {
              current_self_mem:                            rnn_bi_di.output_size, # varies per sli
              previous_synaptic_layer_channel_sl_or_combo: rnn_bi_di.hidden_size, # varies per sli
              previous_synaptic_layer_channel_backward:    0,                     # varies per sli and tci
              next_time_column:                            rnn_bi_di.output_size, # varies per sli and tci
            },
            channel_sl_or_combo: {
              current_self_mem:                            rnn_bi_di.output_size, # varies per sli
              previous_synaptic_layer_channel_sl_or_combo: rnn_bi_di.hidden_size, # varies per sli
              current_forward:                             rnn_bi_di.output_size, # varies per sli
              current_backward:                            rnn_bi_di.output_size, # varies per sli
            },
          },
          # tci: 1
          {
            # channels
            channel_forward: {
              current_self_mem:                            rnn_bi_di.output_size, # varies per sli
              previous_synaptic_layer_channel_sl_or_combo: rnn_bi_di.hidden_size, # varies per sli
              previous_synaptic_layer_channel_forward:     0,                     # rnn_bi_di.output_size, # varies per sli and tci
              previous_time_column:                        rnn_bi_di.output_size, # varies per sli and tci
            },
            channel_backward: {
              current_self_mem:                            rnn_bi_di.output_size, # varies per sli
              previous_synaptic_layer_channel_sl_or_combo: rnn_bi_di.hidden_size, # varies per sli
              previous_synaptic_layer_channel_backward:    0,                     # rnn_bi_di.output_size, # varies per sli and tci
              next_time_column:                            rnn_bi_di.output_size, # varies per sli and tci
            },
            channel_sl_or_combo: {
              current_self_mem:                            rnn_bi_di.output_size, # varies per sli
              previous_synaptic_layer_channel_sl_or_combo: rnn_bi_di.hidden_size, # varies per sli
              current_forward:                             rnn_bi_di.output_size, # varies per sli
              current_backward:                            rnn_bi_di.output_size, # varies per sli
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
          puts "-"*80
          expect(rnn_bi_di.node_input_sizes).to eq(node_input_sizes_expected)
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
          expect(hidden_size_given).to eq(0)
          expect(rnn_bi_di.hidden_size).not_to eq(hidden_size_given)
          expect(rnn_bi_di.hidden_size).to eq(rnn_bi_di.input_size + rnn_bi_di.output_size)
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
              {
                channel_forward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_forward:     0,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_backward:    0,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_forward:     0,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_backward:    0,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
            ],
            [
              {
                channel_forward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     10,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    10,
                  next_time_column:                            10,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     10,
                  previous_time_column:                        10,
                },
                channel_backward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    10,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
            ],
            [
              {
                channel_forward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     10,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    10,
                  next_time_column:                            1,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  current_forward:                             1,
                  current_backward:                            1,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     1,
                  previous_time_column:                        1,
                },
                channel_backward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    1,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 1,
                  current_forward:                             1,
                  current_backward:                            1,
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
              {
                channel_forward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_forward:     0,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_backward:    0,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_forward:     0,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_backward:    0,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_forward:     0,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            0,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  previous_synaptic_layer_channel_backward:    0,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
            ],
            [
              {
                channel_forward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     10,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    10,
                  next_time_column:                            10,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     10,
                  previous_time_column:                        10,
                },
                channel_backward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    10,
                  next_time_column:                            10,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     10,
                  previous_time_column:                        10,
                },
                channel_backward: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    10,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            10,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  current_forward:                             10,
                  current_backward:                            10,
                },
              },
            ],
            [
              {
                channel_forward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     10,
                  previous_time_column:                        0,
                },
                channel_backward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    10,
                  next_time_column:                            1,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 0,
                  current_forward:                             1,
                  current_backward:                            1,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     1,
                  previous_time_column:                        1,
                },
                channel_backward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    1,
                  next_time_column:                            1,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 1,
                  current_forward:                             1,
                  current_backward:                            1,
                },
              },
              {
                channel_forward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_forward:     1,
                  previous_time_column:                        1,
                },
                channel_backward: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 10,
                  previous_synaptic_layer_channel_backward:    1,
                  next_time_column:                            0,
                },
                channel_sl_or_combo: {
                  current_self_mem:                            1,
                  previous_synaptic_layer_channel_sl_or_combo: 1,
                  current_forward:                             1,
                  current_backward:                            1,
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

    #   it "each are of the expected width and height" do
    #     rnn_bi_di.synaptic_layer_indexes.map do |li|
    #       rnn_bi_di.time_col_indexes.map do |ti|
    #         mini_net = rnn_bi_di.mini_net_set[li][ti]

    #         expected_input_size = rnn_bi_di.node_input_sizes[li][ti].values.sum
    #         expect(mini_net.height).to eq(expected_input_size)

    #         expected_output_size = rnn_bi_di.node_output_sizes[li]
    #         expect(mini_net.width).to eq(expected_output_size)
    #       end
    #     end
    #   end
    # end
  end
end
