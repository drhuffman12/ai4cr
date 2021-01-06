require "json"
require "ascii_bar_charter"
require "../../../spec_bench_helper"
require "./../../../../spec/spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnSimple do
  let(steps) { 20 }
  let(scale) { 100 }
  let(noise_delta_t) { 0.1 }
  let(noise_delta_y) { 0.1 }
  let(noise_offset_t) { 0.5 }

  let(sin_data) {
    (0..2*steps).to_a.map do |i|
      theta = (2 * Math::PI * (i / steps.to_f))
      alt = Math.sin(theta)
      ((alt + 1) / (2))
    end
  }

  let(sin_data_offset) {
    (0..2*steps).to_a.map do |i|
      theta = (2 * Math::PI * ((i + noise_offset_t) / steps.to_f))
      alt = Math.sin(theta)
      ((alt + 1) / (2))
    end
  }

  let(sin_data_with_noise) {
    (0..2*steps).to_a.map do |i|
      rnd_noise_t = 2*noise_delta_t*rand - noise_delta_t
      rnd_noise_y = 2*noise_delta_y*rand - noise_delta_y

      i_with_noise = (i + rnd_noise_t)
      theta = (2 * Math::PI * (i_with_noise / steps.to_f))
      alt = Math.sin(theta) + rnd_noise_y
      ((alt + 1) / (2))
    end
  }

  let(sin_data_offset_with_noise) {
    (0..2*steps).to_a.map do |i|
      rnd_noise_t = 2*noise_delta_t*rand - noise_delta_t
      rnd_noise_y = 2*noise_delta_y*rand - noise_delta_y

      i_with_noise = (i + rnd_noise_t)
      theta = (2 * Math::PI * ((i_with_noise + noise_offset_t) / steps.to_f))
      alt = Math.sin(theta) + rnd_noise_y
      ((alt + 1) / (2))
    end
  }

  let(time_col_qty) { 4 }
  let(io_offset) { time_col_qty }

  let(input_size) { 1 }
  let(output_size) { 1 }
  let(hidden_layer_qty) { 1 }

  let(eval_qty) { 3 }

  let(rnn_simple) {
    Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
      io_offset: io_offset,
      time_col_qty: time_col_qty,

      input_size: input_size,
      output_size: output_size,
      hidden_layer_qty: hidden_layer_qty,
    )
  }

  let(split_sin_data) { rnn_simple.split(sin_data, eval_qty) }

  describe "#split" do
    context "given sin_data" do
      context "returns" do
        it "expected type" do
          expect(split_sin_data).to be_an(
            NamedTuple(
              training_data_size: Int32,
              io_pairs_tc_size: Int32,
              io_pairs_qty: Int32,
              io_pairs_indexes: Array(Int32),

              training_qty: Int32,
              eval_qty: Int32,

              io_sets_train: Array(NamedTuple(
                ins: Array(Float64),
                outs: Array(Float64))),
              io_sets_eval: Array(NamedTuple(
                ins: Array(Float64),
                outs: Array(Float64))))
          )
        end

        context "expected value(s)" do
          let(expected_io_pairs_qty) { sin_data.size - io_offset - time_col_qty + 1 }

          it "training_data_size" do
            expect(split_sin_data[:training_data_size]).to eq(sin_data.size)
          end

          it "io_pairs_tc_size" do
            expect(split_sin_data[:io_pairs_tc_size]).to eq(io_offset + time_col_qty)
          end

          it "io_pairs_qty" do
            expect(split_sin_data[:io_pairs_qty]).to eq(expected_io_pairs_qty)
          end

          it "io_pairs_indexes" do
            expected_indexes = (0..(expected_io_pairs_qty - 1)).to_a
            expect(split_sin_data[:io_pairs_indexes]).to eq(expected_indexes)
          end

          it "training_qty" do
            expected_training_qty = split_sin_data[:io_pairs_qty] - eval_qty
            expect(split_sin_data[:training_qty]).to eq(expected_training_qty)
          end

          it "eval_qty" do
            expect(split_sin_data[:eval_qty]).to eq(eval_qty)
          end

          context "io_sets_train" do
            let(io_sets_train) { split_sin_data[:io_sets_train] }

            it "first" do
              expected_io_set = {
                "ins": [
                  0.5,
                  0.6545084971874737,
                  0.7938926261462366,
                  0.9045084971874737,
                ],
                "outs": [
                  0.9755282581475768,
                  1.0,
                  0.9755282581475768,
                  0.9045084971874737,
                ],
              }

              expect(io_sets_train.first).to eq(expected_io_set)
            end

            it "last" do
              expected_io_set = {
                "ins": [
                  0.5000000000000002,
                  0.3454915028125265,
                  0.2061073738537636,
                  0.09549150281252639,
                ],
                "outs": [
                  0.02447174185242329,
                  0.0,
                  0.024471741852423123,
                  0.09549150281252616,
                ],
              }

              expect(io_sets_train.last).to eq(expected_io_set)
            end

            it "size" do
              expect(io_sets_train.size).to eq(split_sin_data[:io_pairs_qty] - eval_qty)
            end
          end

          context "io_sets_eval" do
            let(io_sets_eval) { split_sin_data[:io_sets_eval] }

            it "first" do
              expected_io_set = {
                "ins": [
                  0.3454915028125265,
                  0.2061073738537636,
                  0.09549150281252639,
                  0.02447174185242329,
                ],
                "outs": [
                  0.0,
                  0.024471741852423123,
                  0.09549150281252616,
                  0.20610737385376326,
                ],
              }

              expect(io_sets_eval.first).to eq(expected_io_set)
            end

            it "last" do
              expected_io_set = {
                "ins": [
                  0.09549150281252639,
                  0.02447174185242329,
                  0.0,
                  0.024471741852423123,
                ],
                "outs": [
                  0.09549150281252616,
                  0.20610737385376326,
                  0.34549150281252605,
                  0.4999999999999998,
                ],
              }

              expect(io_sets_eval.last).to eq(expected_io_set)
            end

            it "size" do
              expect(io_sets_eval.size).to eq(eval_qty)
            end
          end
        end
      end
    end
  end

  describe "#float_to_state" do
    let(to_min_i) { 0 }
    let(to_max_i) { 10 }
    let(from_min) { 0.0 }

    let(input_size) { to_max_i - to_min_i + 1 }
    let(output_size) { to_max_i - to_min_i + 1 }
    let(hidden_layer_qty) { 1 }

    pending "foo" do
      # to_min_i = 0
      # to_max_i = 10
      # from_min = 0.0

      # 'sin_data' already mapped to range 0..1
      value_states = rnn_simple.float_to_state(values: sin_data, to_min_i: to_min_i, to_max_i: to_max_i, from_min: from_min)

      puts
      puts "value_states: #{value_states.pretty_inspect}"

      split_values = rnn_simple.split(value_states, eval_qty: 3)

      puts
      puts "split_values: #{split_values.pretty_inspect}"
      puts

      puts
      puts "node_input_sizes:
      #{rnn_simple.node_input_sizes.pretty_inspect}"
      puts

      # results = rnn_simple.train_in_sequence(split_values)
      # puts "vvvv"
      # puts "results: #{results.pretty_inspect}"
      # puts "----"
    end
  end
end
