require "./../../../../../spec_helper"
require "./../../../../../spectator_helper"

Spectator.describe Ai4cr::NeuralNetwork::Cmn::RnnConcerns::TrainAndAdjust do
  let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

  let(input_set_given) {
    [
      [0.1, 0.2],
      [0.3, 0.4],
    ]
  }
  let(delta_1_thousandths) { 0.000_1 }

  describe "#train" do
    let(expected_outputs_guessed_before) { [[0.0], [0.0]] }
    let(expected_outputs_trained) { [[0.4], [0.6]] }

    context "with hard-coded weights" do
      let(expected_outputs_guessed) {
        [[0.14193], [0.0]]
      }
      let(hard_coded_weights) {
        [
          [
            [
              [-0.4, -0.3, -0.2],
              [-0.1, 0.05, 0.1],
              [0.2, 0.3, 0.4],
            ],
            [
              [-0.41, -0.31, -0.21],
              [-0.11, 0.06, 0.11],
              [0.21, 0.31, 0.41],
              [-0.42, -0.32, -0.22],
              [-0.12, 0.07, 0.12],
              [0.22, 0.32, 0.42],
            ],
          ],
          [
            [
              [-0.23],
              [0.13],
              [0.33],
            ],
            [
              [-0.44],
              [-0.24],
              [0.24],
              [0.44],
            ],
          ],
        ]
      }

      before_each do
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            # NOTE: We 'clone' the values so that our tests don't affect the original 'hard_coded_weights'!
            rnn_simple.mini_net_set[li][ti].weights = hard_coded_weights[li][ti].clone
          end
        end
      end

      context "before" do
        it "outputs_guessed start off all zero's" do
          expect(rnn_simple.outputs_guessed).to eq(expected_outputs_guessed_before)
        end
      end

      context "during" do
        pending "calls 'eval(input_set_given)'" do
          # TODO: un-pend after '...to receive...' is fixed
          expect(rnn_simple).to receive(:eval).with(input_set_given)

          rnn_simple.train(input_set_given, expected_outputs_trained)
        end
      end

      context "after" do
        let(expected_error_total) { 0.016805353667424198 }

        it "guesses expected outputs" do
          rnn_simple.train(input_set_given, expected_outputs_trained)

          assert_approximate_equality_of_nested_list(expected_outputs_guessed, rnn_simple.outputs_guessed)
        end

        it "calculates output_deltas" do
          before_all_mini_net_output_deltas = rnn_simple.all_mini_net_output_deltas.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_output_deltas = rnn_simple.all_mini_net_output_deltas.clone

          assert_approximate_inequality_of_nested_list(before_all_mini_net_output_deltas, after_all_mini_net_output_deltas, delta_1_thousandths)
        end

        it "calculates input_deltas" do
          before_all_mini_net_input_deltas = rnn_simple.all_mini_net_input_deltas.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_input_deltas = rnn_simple.all_mini_net_input_deltas.clone

          assert_approximate_inequality_of_nested_list(after_all_mini_net_input_deltas, before_all_mini_net_input_deltas, delta_1_thousandths)
        end

        it "adjusts weights" do
          rnn_simple.train(input_set_given, expected_outputs_trained)

          assert_approximate_inequality_of_nested_list(hard_coded_weights, rnn_simple.all_mini_net_weights, delta_1_thousandths)
        end

        it "caches last_changes" do
          before_all_mini_net_last_changes = rnn_simple.all_mini_net_last_changes.clone

          rnn_simple.train(input_set_given, expected_outputs_trained)

          after_all_mini_net_last_changes = rnn_simple.all_mini_net_last_changes.clone

          assert_approximate_inequality_of_nested_list(before_all_mini_net_last_changes, after_all_mini_net_last_changes, delta_1_thousandths)
        end

        it "returns expected error_total" do
          error_total = rnn_simple.train(input_set_given, expected_outputs_trained)

          expect(error_total).to eq(expected_error_total)
        end
      end
    end
  end

  describe "#all_mini_net_outputs" do
    context "with hard-coded weights" do
      let(rnn_simple) { Ai4cr::NeuralNetwork::Cmn::RnnSimple.new }

      let(input_set_given) {
        [
          [0.1, 0.2],
          [0.3, 0.4],
        ]
      }

      let(expected_all_mini_net_outputs_before) {
        [
          [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
          ],
          [
            [0.0],
            [0.0],
          ],
        ]
      }

      let(expected_all_mini_net_outputs_after) {
        # TODO: manually verify calc's.
        #   For now, we'll assume accuracy and move onto
        #   verifying via some training sessions with
        #   some 'real data'
        [
          [
            [0.14, 0.27, 0.4],
            [0.0, 0.17099999999999999, 0.42200000000000004],
          ],
          [
            [0.119],
            [0.09780000000000003],
          ],
        ]
      }

      before_each do
        weights = [
          [
            [
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
            ],
            [
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
              [-0.4, -0.3, -0.2],
              [-0.1, 0.0, 0.1],
              [0.2, 0.3, 0.4],
            ],
          ],
          [
            [
              [-0.2],
              [0.1],
              [0.3],
            ],
            [
              [-0.4],
              [-0.2],
              [0.2],
              [0.4],
            ],
          ],
        ]
        rnn_simple.synaptic_layer_indexes.map do |li|
          rnn_simple.time_col_indexes.map do |ti|
            rnn_simple.mini_net_set[li][ti].weights = weights[li][ti]
          end
        end
      end

      context "before #eval" do
        it "returns all-zero outputs" do
          expect(rnn_simple.all_mini_net_outputs).to eq(expected_all_mini_net_outputs_before)
        end
      end

      context "after #eval" do
        it "returns expected non-zero outputs" do
          rnn_simple.eval(input_set_given)

          assert_approximate_equality_of_nested_list(expected_all_mini_net_outputs_after, rnn_simple.all_mini_net_outputs, delta_1_thousandths**3)
        end
      end
    end
  end
end
