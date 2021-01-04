require "json"
require "ascii_bar_charter"
require "../../../spec_bench_helper"
require "./../../../../spec/spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnSimple do
  let(steps) { 20 }
  let(scale) { 100 }
  let(noise_delta) { 0.1 }
  let(sin_data) {
    (0..2*steps).to_a.map do |i|
      theta = (2 * Math::PI * (i / steps.to_f))
      alt = Math.sin(theta) # .round(4)
      ((alt + 1) / (2))
    end
  }

  let(sin_data_with_noise) {
    (0..2*steps).to_a.map do |i|
      i_with_noise = (i + 2*noise_delta*rand - noise_delta)
      theta = (2 * Math::PI * (i_with_noise / steps.to_f))
      alt = Math.sin(theta) # .round(4)
      ((alt + 1) / (2))
    end
  }

  let(time_col_qty) { 4 }
  let(io_offset) { time_col_qty }

  let(input_size) { 1 }
  let(output_size) { 1 }

  let(eval_qty) { 3 }

  let(rnn_simple) {
    Ai4cr::NeuralNetwork::Cmn::RnnSimple.new(
      io_offset: io_offset,
      time_col_qty: time_col_qty,

      input_size: 1,
      output_size: 1,
      hidden_layer_qty: 1,
    )
  }

  let(split_sin_data) { rnn_simple.split(sin_data, eval_qty) }

  # let(split_sin_data_with_noise) {
  #   rnn_simple.split_as_all_eval(sin_data_with_noise)
  # }

  describe "#split" do
    context "given sin_data" do
      context "returns" do
        it "expected type" do
          # puts "split_sin_data: #{split_sin_data.to_pretty_json}"

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
            expect(split_sin_data[:io_pairs_tc_size]).to eq(io_offset + time_col_qty) # assuming inputs are not
          end

          it "io_pairs_qty" do
            expect(split_sin_data[:io_pairs_qty]).to eq(expected_io_pairs_qty)
          end

          it "io_pairs_indexes" do
            expected_indexes = (0..(expected_io_pairs_qty - 1)).to_a
            expect(split_sin_data[:io_pairs_indexes]).to eq(expected_indexes)
          end

          # it "io_sets_size" do
          #   # expected_io_sets_size = io_pairs_qty # (expected_io_pairs_qty - 1) - eval_qty
          #   expect(split_sin_data[:io_sets_size]).to eq(split_sin_data[:io_pairs_qty])
          # end

          it "training_qty" do
            expected_training_qty = split_sin_data[:io_pairs_qty] - eval_qty # (expected_io_pairs_qty - 1) - eval_qty
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

  describe "#train_in_sequence" do
    context "given sin_data" do
      # TODO: tests for 'train_in_sequence'

      context "given sin_data_with_noise" do
        # TODO: tests for 'train_in_sequence'
      end
    end
  end
end
